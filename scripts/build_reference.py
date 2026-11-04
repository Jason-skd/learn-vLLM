"""Generate reference tensors with Transformers"""

import json
import platform
from pathlib import Path
from typing import Literal

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# reference/ dir at project root; both config.json and outputs.pt live here
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"
REFERENCE_SCHEMA_VERSION = 2
ReferenceMode = Literal["full", "cached"]
REFERENCE_MODES: tuple[ReferenceMode, ...] = ("full", "cached")


@torch.inference_mode()  # inference mode: no grad, less memory, faster, same results
def run_case(model, tokenizer, case, config, cache_mode: ReferenceMode):
    """Generate one greedy reference using the selected attention path."""
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
    step_input = ids  # the full prompt during prefill and new token afterward
    past_key_values = None
    # sequence:  [1, L]  grows to [1, L + step] as new tokens are appended
    # attention: [1, L]  grows in lockstep, always same length as sequence

    generated, step_logits = [], []
    # generated:   list, length N at the end, each item [1, 1]
    # step_logits: list, length N at the end, each item [1, V]

    eos = model.generation_config.eos_token_id
    # eos may be None / int / list[int]; normalize to a set for O(1) lookup
    eos_ids = set() if eos is None else set(eos if isinstance(eos, list) else [eos])

    stop_reason = "max_new_tokens"  # default

    # ---------- 3. generate ----------
    use_cache = cache_mode == "cached"
    for step in range(config["generation"]["max_new_tokens"]):
        # at iteration `step`, sequence length is L + step
        output = model(
            input_ids=step_input if use_cache else sequence,
            attention_mask=attention,
            past_key_values=past_key_values,
            use_cache=use_cache,
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
        if use_cache:
            past_key_values = output.past_key_values
            step_input = token
            # step_input: [1, 1]  only the new token is processed

    # ---------- 4. pack ----------
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


def build_artifact(config):
    """Build full-context and cached references from one official model."""
    # ---------- 1. validate config ----------
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
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "reference_modes": list(REFERENCE_MODES),
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
            "cache_implementation": "DynamicCache",
        },
        "cases": {},
    }

    # ---------- 4. run all cases in both modes ----------
    for case in config["cases"]:
        results = {
            mode: run_case(model, tokenizer, case, config, mode)
            for mode in REFERENCE_MODES
        }
        full = results["full"]
        artifact["cases"][case["id"]] = {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "modes": results,
        }
        for mode, result in results.items():
            print(f"{case['id']}/{mode}: {result['generated_text']!r}")

    return artifact


def main():
    config = json.loads((REFERENCE_DIR / "config.json").read_text(encoding="utf-8"))
    artifact = build_artifact(config)
    destination = REFERENCE_DIR / "outputs.pt"
    torch.save(artifact, destination)  # pickle-based; handles dict + tensors
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
