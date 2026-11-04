"""Compare local generation with saved reference tensors."""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from learn_vllm.generation import greedy_generate
from learn_vllm.model import Qwen3ForCausalLM

pytestmark = pytest.mark.integration

# reference/ dir at project root; config.json and outputs.pt live here
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"
REFERENCE_SCHEMA_VERSION = 2
REFERENCE_MODES = ("full", "cached")


def load_config():
    """Load the tracked configuration shared by reference and integration tests."""
    return json.loads((REFERENCE_DIR / "config.json").read_text(encoding="utf-8"))


def pytest_generate_tests(metafunc):
    """Expose each configured reference case as an independently reported test."""
    if "case" in metafunc.fixturenames:
        cases = load_config()["cases"]
        metafunc.parametrize("case", cases, ids=[case["id"] for case in cases])
    if "mode" in metafunc.fixturenames:
        metafunc.parametrize("mode", REFERENCE_MODES)


def resolve_model_id(config, reference_config):
    """Prefer an available local model path over a remote Hub identifier."""
    current_id = config["model"]["id"]
    artifact_id = reference_config["model"]["id"]
    for model_id in (current_id, artifact_id):
        path = Path(model_id)
        if path.is_absolute() and path.is_dir():
            return model_id
    return current_id


@pytest.fixture(scope="module")
def reference_artifact():
    """Load the reference output and verify it matches the current configuration."""
    artifact_path = REFERENCE_DIR / "outputs.pt"
    if not artifact_path.exists():
        pytest.skip(
            f"{artifact_path} does not exist; run scripts/build_reference.py first"
        )

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported reference artifact schema version")
    if artifact.get("reference_modes") != list(REFERENCE_MODES):
        raise ValueError("Reference artifact does not contain both generation modes")

    config = load_config()
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

    return config, artifact, artifact_path


@pytest.fixture(scope="module")
def runtime(reference_artifact):
    """Load one local model and tokenizer for all reference cases."""
    config, artifact, _ = reference_artifact
    mc = config["model"]
    model_id = resolve_model_id(config, artifact["config"])

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

    return model, tokenizer, eos_ids


@torch.inference_mode()
def run_case(model, tokenizer, case, config, case_reference, mode, eos_token_ids):
    """Assert tokenization, logits, generated IDs, stop reason and decoded text."""
    # shape notation: B=1 (batch), L=input tokens, N=generated tokens, V=vocab size
    expected = case_reference["modes"][mode]

    # ---------- 1. tokenize ----------
    inputs = tokenizer(
        case["text"],
        add_special_tokens=config["tokenization"]["add_special_tokens"],
        return_tensors="pt",
    ).to(config["model"]["device"])
    ids = inputs["input_ids"]
    # ids: [1, L] int64
    torch.testing.assert_close(ids.cpu(), case_reference["input_ids"], rtol=0, atol=0)

    # ---------- 2. generate ----------
    cache = None
    if mode == "cached":
        cache = model.create_kv_cache(
            ids.shape[1] + config["generation"]["max_new_tokens"]
        )
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
    assert result.stop_reason == expected["stop_reason"], (
        f"Stop reason differs: {result.stop_reason!r} != {expected['stop_reason']!r}"
    )

    generated_text = tokenizer.decode(result.generated_ids[0], skip_special_tokens=True)
    assert generated_text == expected["generated_text"], (
        f"Generated text differs:\n{generated_text!r}\n{expected['generated_text']!r}"
    )

    return generated_text


def test_generation_matches_reference(runtime, reference_artifact, case, mode):
    """Compare one generation mode with its matching Transformers output."""
    config, artifact, artifact_path = reference_artifact
    case_id = case["id"]
    if case_id not in artifact["cases"]:
        pytest.fail(f"Reference artifact has no case: {case_id}")

    model, tokenizer, eos_ids = runtime
    generated_text = run_case(
        model,
        tokenizer,
        case,
        config,
        artifact["cases"][case_id],
        mode,
        eos_ids,
    )
    print(f"{case_id}/{mode}: {generated_text!r} (matched {artifact_path})")
