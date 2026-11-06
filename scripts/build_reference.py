"""Generate reference tensors with Transformers"""

import json
import platform
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# reference/ dir at project root; both config.json and outputs.pt live here
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"


@torch.inference_mode()  # inference mode: no grad, less memory, faster, same results
def run_case(model, tokenizer, case, config):
    # shape notation: B=1 (batch), L=input tokens, N=generated tokens, V=vocab size

    # ---------- 1. tokenize ----------
    inputs = tokenizer(
        case["text"],
        add_special_tokens=config["tokenization"]["add_special_tokens"],
        return_tensors="pt",
    ).to(config["model"]["device"])
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    # ids:  [1, L]  int64
    # mask: [1, L]  int64 (no padding in this script, so all ones)
    if ids.shape[1] == 0:
        raise ValueError(f"Empty token sequence: {case['id']}")

    # ---------- 2. prepare vars ----------
    sequence, attention = ids, mask
    # sequence:  [1, L]  grows to [1, L + step] as new tokens are appended
    # attention: [1, L]  grows in lockstep, always same length as sequence

    generated, step_logits = [], []
    # generated:   list, length N at the end, each item [1, 1]
    # step_logits: list, length N at the end, each item [1, V]
    prefill_logits: torch.Tensor | None = None

    eos = model.generation_config.eos_token_id
    # eos may be None / int / list[int]; normalize to a set for O(1) lookup
    eos_ids = set() if eos is None else set(eos if isinstance(eos, list) else [eos])

    stop_reason = "max_new_tokens"  # default

    # ---------- 3. generate ----------
    for step in range(config["generation"]["max_new_tokens"]):
        # at iteration `step`, sequence length is L + step
        output = model(
            input_ids=sequence,
            attention_mask=attention,
            use_cache=False,
        )
        # output: CausalLMOutputWithPast
        # output.logits: [1, L + step, V]

        if step == 0:
            # save the full prefill logits once, as one of the reference tensors
            prefill_logits = output.logits.float().cpu().clone()
            # prefill_logits: [1, L, V]

        logits = output.logits[:, -1, :]
        # logits: [1, V]

        token = logits.argmax(dim=-1, keepdim=True)
        # token: [1, 1]  int64, the greedily chosen next token

        step_logits.append(logits.float().cpu().clone())
        generated.append(token.cpu())

        if config["generation"]["stop_on_eos"] and token.item() in eos_ids:
            stop_reason = "eos"
            break

        sequence = torch.cat((sequence, token), dim=1)
        # sequence: [1, L + step + 1]  append new token along the seq dim

        attention = torch.cat((attention, torch.ones_like(token)), dim=1)
        # attention: [1, L + step + 1]  append one more 1

    # ---------- 4. pack ----------
    # main() validates that the generation limit is positive.
    assert prefill_logits is not None
    new_ids = torch.cat(generated, dim=1)
    # new_ids: [1, N]

    return {
        "input_ids": ids.cpu(),
        "attention_mask": mask.cpu(),
        "prefill_logits": prefill_logits,
        "generated_ids": new_ids,
        "generation_logits": torch.stack(step_logits, dim=1),
        "generated_text": tokenizer.decode(new_ids[0], skip_special_tokens=True),
        "stop_reason": stop_reason,
    }


def main():
    # ---------- 1. load & validate config ----------
    config = json.loads((REFERENCE_DIR / "config.json").read_text(encoding="utf-8"))

    if config["generation"]["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be positive")

    case_ids = [case["id"] for case in config["cases"]]
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Cases must be nonempty and have unique IDs")

    # ---------- 2. load model & tokenizer ----------
    mc = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(mc["id"], revision=mc["revision"])
    model = (
        AutoModelForCausalLM.from_pretrained(
            mc["id"],
            revision=mc["revision"],
            dtype=getattr(torch, mc["dtype"]),
            attn_implementation="eager",  # plainest attention impl; most reproducible
        )
        .to(mc["device"])
        .eval()
    )  # .eval(): disable dropout, deterministic results

    # ---------- 3. build artifact ----------
    artifact = {
        "schema_version": 1,
        "config": config,
        "runtime": {
            # environment fingerprint
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "device": mc["device"],
            "dtype": mc["dtype"],
            "attention_implementation": "eager",
        },
        "cases": {},
    }

    # ---------- 4. run all cases ----------
    for case in config["cases"]:
        result = run_case(model, tokenizer, case, config)
        artifact["cases"][case["id"]] = result
        print(f"{case['id']}: {result['generated_text']!r}")

    # ---------- 5. save ----------
    destination = REFERENCE_DIR / "outputs.pt"
    torch.save(artifact, destination)  # pickle-based; handles dict + tensors
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
