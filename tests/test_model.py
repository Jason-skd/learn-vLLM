import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM as ReferenceModel,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from learn_vllm.model import Qwen3ForCausalLM, RotaryEmbedding


def make_config(**kwargs):
    """Builds a small Qwen3 configuration with optional overrides for each test."""
    return Qwen3Config(
        vocab_size=67,
        hidden_size=24,
        intermediate_size=40,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,  # Hq * D differs from hidden_size, as in Qwen3-0.6B.
        **kwargs,
    )


def test_rope_frequency_rounding_matches_reference():
    """Checks that inverse frequencies match the reference without rounding drift."""
    # ---------- 1. build RoPE modules with the real model dimensions ----------
    # Equivalent formulas round differently at the actual Qwen3-0.6B dimensions.
    config = Qwen3Config(
        head_dim=128,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    )
    rotary = RotaryEmbedding(128, 1000000.0)
    reference = Qwen3RotaryEmbedding(config)

    # ---------- 2. require exact equality of the inverse frequencies ----------
    torch.testing.assert_close(rotary.inv_freq, reference.inv_freq, rtol=0, atol=0)


@pytest.mark.parametrize("length", [1, 7])
@pytest.mark.parametrize("tied,bias", [(True, False), (False, True)])
@torch.inference_mode()
def test_forward_matches_reference(length, tied, bias):
    """Compares intermediate states and logits after loading identical weights."""
    # ---------- 1. build both models and load the same weights ----------
    torch.manual_seed(42)
    config = make_config(tie_word_embeddings=tied, attention_bias=bias)
    config._attn_implementation = "eager"
    reference = ReferenceModel(config).eval()
    model = Qwen3ForCausalLM(config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.randint(config.vocab_size, (1, length))
    # ids: [1, L], shared by both models

    # ---------- 2. capture outputs at the main assembly boundaries ----------
    # Compare the first attention, first decoder layer and final normalization.
    actual, expected = {}, {}
    handles = []
    for name in ("model.layers.0.self_attn", "model.layers.0", "model.norm"):
        for network, outputs in ((model, actual), (reference, expected)):

            def capture(module, args, output, name=name, outputs=outputs):
                # Bind each hook's name and destination when it is created.
                # Reference attention returns a tuple; our modules return tensors.
                outputs[name] = output[0] if isinstance(output, tuple) else output

            handles.append(network.get_submodule(name).register_forward_hook(capture))

    # ---------- 3. run full-context forward passes and remove hooks ----------
    try:
        logits = model(ids)
        reference_logits = reference(ids, use_cache=False).logits
        # logits/reference_logits: [1, L, vocab_size]
    finally:
        # Remove hooks even if a forward pass fails.
        for handle in handles:
            handle.remove()

    # ---------- 4. compare intermediate states and all token logits ----------
    for name, output in actual.items():
        torch.testing.assert_close(output, expected[name], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(logits, reference_logits, rtol=1e-4, atol=1e-5)

    # ---------- 5. verify whether embeddings and the output head share weights ----------
    # Parameter identity checks sharing, not just equality of the stored values.
    assert (model.lm_head.weight is model.model.embed_tokens.weight) == tied


@torch.inference_mode()
def test_future_tokens_do_not_change_prefix_logits():
    """Checks that changing or removing future tokens leaves prefix logits unchanged."""
    # ---------- 1. prepare sequences with the same three-token prefix ----------
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(make_config()).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    changed = torch.tensor([[1, 2, 3, 8, 9]])
    # ids/changed: [1, 5]; only the final two tokens differ

    # ---------- 2. verify that future token values cannot affect the prefix ----------
    torch.testing.assert_close(model(ids)[:, :3], model(changed)[:, :3])

    # ---------- 3. verify that removing future tokens preserves the prefix ----------
    torch.testing.assert_close(model(ids)[:, :3], model(ids[:, :3]))


@torch.inference_mode()
def test_cached_decode_matches_full_forward():
    """Checks that continuous cache reuse produces the uncached decode logits."""
    # ---------- 1. build a model and prefill a three-token prompt ----------
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(make_config()).eval()
    prompt = torch.tensor([[1, 2, 3]])
    cache = model.create_kv_cache(max_seq_len=5)

    cached_prefill = model(prompt, cache=cache)
    full_prefill = model(prompt)
    # cached_prefill/full_prefill: [1, 3, V]

    torch.testing.assert_close(cached_prefill, full_prefill, rtol=1e-4, atol=1e-5)
    assert cache.length == 3

    # ---------- 2. decode one token at each absolute position ----------
    next_token = torch.tensor([[4]])
    cached_logits = model(next_token, cache=cache)
    full_logits = model(torch.tensor([[1, 2, 3, 4]]))[:, -1:, :]
    # cached_logits/full_logits: [1, 1, V]

    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-4, atol=1e-5)
    assert cache.length == 4

    next_token = torch.tensor([[5]])
    cached_logits = model(next_token, cache=cache)
    full_logits = model(torch.tensor([[1, 2, 3, 4, 5]]))[:, -1:, :]

    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-4, atol=1e-5)
    assert cache.length == 5


@pytest.mark.parametrize("shape", [(0,), (1, 0), (2, 3), (1, 2, 3)])
def test_rejects_unsupported_input_shapes(shape):
    """Rejects empty sequences, multiple requests and tensors with the wrong rank."""
    model = Qwen3ForCausalLM(make_config())
    with pytest.raises(ValueError, match="input_ids"):
        model(torch.zeros(shape, dtype=torch.long))


@pytest.mark.parametrize(
    "options,message",
    [
        ({"hidden_act": "relu"}, "SiLU"),
        ({"attention_dropout": 0.1}, "dropout"),
        ({"layer_types": ["sliding_attention"] * 2}, "full attention"),
        ({"rope_parameters": {"rope_type": "linear", "factor": 2.0}}, "default RoPE"),
    ],
)
def test_rejects_unsupported_configurations(options, message):
    """Rejects configurations whose behavior the current forward cannot reproduce."""
    with pytest.raises(ValueError, match=message):
        Qwen3ForCausalLM(make_config(**options))
