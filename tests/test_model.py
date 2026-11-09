import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM as ReferenceModel,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from learn_vllm.model import Attention, Qwen3ForCausalLM, RotaryEmbedding


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
        logits, _ = model(ids)
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
    logits, _ = model(ids)
    changed_logits, _ = model(changed)
    torch.testing.assert_close(logits[:, :3], changed_logits[:, :3])

    # ---------- 3. verify that removing future tokens preserves the prefix ----------
    prefix_logits, _ = model(ids[:, :3])
    torch.testing.assert_close(logits[:, :3], prefix_logits)


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


@torch.inference_mode()
def test_attention_cached_suffix_matches_full_sequence():
    # ---------- 1. prepare a small attention module and a RoPE module ----------
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        attention_bias=False,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    attention = Attention(config).eval()
    rope = RotaryEmbedding(head_dim=4, theta=10000.0)

    # --------- 2. prepare a random input sequence and compute the full attention ----------
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 5, 16, generator=generator, dtype=torch.float32)

    positions = torch.arange(5).unsqueeze(0)
    cos, sin = rope(positions, dtype=x.dtype)

    full_output, _ = attention(x, cos, sin)

    # --------- 3. compute the attention in two parts, caching the first part ----------
    _, cache = attention(
        x[:, :3],
        cos[:, :3],
        sin[:, :3],
    )
    suffix_output, _ = attention(
        x[:, 3:],
        cos[:, 3:],
        sin[:, 3:],
        past_key_value=cache,
    )

    # --------- 4. verify that the cached suffix matches the full sequence ----------
    torch.testing.assert_close(
        suffix_output,
        full_output[:, 3:],
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("prefix_length", [1, 3, 4])
@torch.inference_mode()
def test_model_cached_suffix_matches_full_sequence(prefix_length):
    """Checks cached logits and cache growth across all decoder layers."""
    # ---------- 1. prepare a small model and a fixed token sequence ----------
    torch.manual_seed(42)
    config = make_config()
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])

    # ---------- 2. compute the full sequence and the cached prefix ----------
    full_logits, _ = model(ids)
    prefix_logits, cache = model(ids[:, :prefix_length])

    assert len(cache) == config.num_hidden_layers
    for k, v in cache:
        assert (
            k.shape
            == v.shape
            == (
                1,
                config.num_key_value_heads,
                prefix_length,
                config.head_dim,
            )
        )

    # ---------- 3. process only the suffix using the prefix cache ----------
    suffix_logits, updated_cache = model(
        ids[:, prefix_length:],
        past_key_values=cache,
    )

    # ---------- 4. compare logits and verify that every cache grew ----------
    torch.testing.assert_close(
        prefix_logits,
        full_logits[:, :prefix_length],
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        suffix_logits,
        full_logits[:, prefix_length:],
        rtol=1e-4,
        atol=1e-5,
    )

    for k, v in updated_cache:
        assert (
            k.shape
            == v.shape
            == (
                1,
                config.num_key_value_heads,
                ids.shape[1],
                config.head_dim,
            )
        )

    # Concatenation creates new tensors; the supplied prefix cache stays usable.
    for k, v in cache:
        assert k.shape[2] == v.shape[2] == prefix_length
