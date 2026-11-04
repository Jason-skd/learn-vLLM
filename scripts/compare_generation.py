"""Compare local single-request generation with saved reference tensors."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from learn_vllm.generation import greedy_generate
from learn_vllm.model import Qwen3ForCausalLM

# reference/ dir at project root; config.json and outputs.pt live here
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"


def resolve_model_id(config, reference_config):
    """Prefers an available local model path over a remote Hub identifier."""
    current_id = config["model"]["id"]
    artifact_id = reference_config["model"]["id"]
    for model_id in (current_id, artifact_id):
        path = Path(model_id)
        if path.is_absolute() and path.is_dir():
            return model_id
    return current_id


@torch.inference_mode()
def run_case(model, tokenizer, case, config, expected, eos_token_ids):
    # shape notation: B=1 (batch), L=input tokens, N=generated tokens, V=vocab size

    # ---------- 1. tokenize ----------
    inputs = tokenizer(
        case["text"],
        add_special_tokens=config["tokenization"]["add_special_tokens"],
        return_tensors="pt",
    ).to(config["model"]["device"])
    ids = inputs["input_ids"]
    # ids: [1, L] int64
    torch.testing.assert_close(ids.cpu(), expected["input_ids"], rtol=0, atol=0)

    # ---------- 2. generate ----------
    cache = model.create_kv_cache(ids.shape[1] + config["generation"]["max_new_tokens"])
    result = greedy_generate(
        model,
        ids,
        max_new_tokens=config["generation"]["max_new_tokens"],
        eos_token_ids=eos_token_ids,
        cache=cache,
    )

    # ---------- 3. compare ----------
    torch.testing.assert_close(
        result.prefill_logits,
        expected["prefill_logits"],
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        result.generation_logits,
        expected["generation_logits"],
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        result.generated_ids,
        expected["generated_ids"],
        rtol=0,
        atol=0,
    )
    if result.stop_reason != expected["stop_reason"]:
        raise AssertionError(
            f"Stop reason differs: {result.stop_reason!r} != "
            f"{expected['stop_reason']!r}"
        )

    generated_text = tokenizer.decode(result.generated_ids[0], skip_special_tokens=True)
    if generated_text != expected["generated_text"]:
        raise AssertionError(
            f"Generated text differs:\n{generated_text!r}\n"
            f"{expected['generated_text']!r}"
        )

    return generated_text


def main() -> None:
    # ---------- 1. load and validate the reference artifact ----------
    artifact_path = REFERENCE_DIR / "outputs.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"{artifact_path} does not exist; run scripts/build_reference.py first"
        )
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != 1:
        raise ValueError("Unsupported reference artifact schema version")

    config = json.loads((REFERENCE_DIR / "config.json").read_text(encoding="utf-8"))
    reference_config = artifact["config"]
    for key in ("tokenization", "generation", "cases"):
        if config[key] != reference_config[key]:
            raise ValueError(
                f"{key} differs from the reference artifact; "
                "regenerate reference/outputs.pt"
            )
    if config["model"]["dtype"] != artifact["runtime"]["dtype"]:
        raise ValueError("Model dtype differs from the reference artifact")
    if config["model"]["device"] != artifact["runtime"]["device"]:
        raise ValueError("Model device differs from the reference artifact")

    mc = config["model"]
    model_id = resolve_model_id(config, reference_config)

    # ---------- 2. load model, tokenizer and EOS configuration ----------
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=mc["revision"])
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=mc["revision"],
        dtype=getattr(torch, mc["dtype"]),
        attn_implementation="eager",
    ).eval()

    eos = reference_model.generation_config.eos_token_id
    eos_ids = [] if eos is None else (eos if isinstance(eos, list) else [eos])
    if not config["generation"]["stop_on_eos"]:
        eos_ids = []

    model = Qwen3ForCausalLM(reference_model.config)
    model.load_state_dict(reference_model.state_dict(), strict=True)
    model = model.to(device=mc["device"], dtype=getattr(torch, mc["dtype"])).eval()
    del reference_model

    # ---------- 3. compare all reference cases ----------
    for case in config["cases"]:
        case_id = case["id"]
        if case_id not in artifact["cases"]:
            raise KeyError(f"Reference artifact has no case: {case_id}")

        generated_text = run_case(
            model,
            tokenizer,
            case,
            config,
            artifact["cases"][case_id],
            eos_ids,
        )
        print(f"{case_id}: {generated_text!r}")

    print(f"All {len(config['cases'])} cases match {artifact_path}")


if __name__ == "__main__":
    main()
