import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM as ReferenceModel,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from learn_vllm.model import Qwen3ForCausalLM, RotaryEmbedding, causal_attention


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
    cache = model.create_kv_cache(max_seq_len=5, batch_size=1)

    cached_prefill = model(prompt, cache=cache)
    full_prefill = model(prompt)
    # cached_prefill/full_prefill: [1, 3, V]

    torch.testing.assert_close(cached_prefill, full_prefill, rtol=1e-4, atol=1e-5)
    assert cache.lengths.tolist() == [3]

    # ---------- 2. decode one token at each absolute position ----------
    next_token = torch.tensor([[4]])
    cached_logits = model(next_token, cache=cache)
    full_logits = model(torch.tensor([[1, 2, 3, 4]]))[:, -1:, :]
    # cached_logits/full_logits: [1, 1, V]

    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-4, atol=1e-5)
    assert cache.lengths.tolist() == [4]

    next_token = torch.tensor([[5]])
    cached_logits = model(next_token, cache=cache)
    full_logits = model(torch.tensor([[1, 2, 3, 4, 5]]))[:, -1:, :]

    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-4, atol=1e-5)
    assert cache.lengths.tolist() == [5]


@torch.inference_mode()
def test_batched_cached_decode_matches_individual_forward():
    """Checks variable prefill lengths and cache positions across a batch."""
    # ---------- 1. prefill two different-length prompts in one cache ----------
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(make_config()).eval()
    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 0],
        ]
    )
    token_mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
        ]
    )
    cache = model.create_kv_cache(max_seq_len=6, batch_size=2)

    cached_prefill = model(
        input_ids,
        token_mask=token_mask,
        cache=cache,
    )
    assert cache.lengths.tolist() == [3, 2]

    for row, length in enumerate((3, 2)):
        full = model(input_ids[row : row + 1, :length])
        torch.testing.assert_close(
            cached_prefill[row : row + 1, :length],
            full,
            rtol=1e-4,
            atol=1e-5,
        )

    # ---------- 2. decode one token for both requests ----------
    next_tokens = torch.tensor([[6], [7]])
    cached_logits = model(next_tokens, cache=cache)
    assert cache.lengths.tolist() == [4, 3]

    expected_0 = model(torch.tensor([[1, 2, 3, 6]]))[:, -1:, :]
    expected_1 = model(torch.tensor([[4, 5, 7]]))[:, -1:, :]

    torch.testing.assert_close(
        cached_logits[0:1],
        expected_0,
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        cached_logits[1:2],
        expected_1,
        rtol=1e-4,
        atol=1e-5,
    )

    # ---------- 3. finish request one while request zero keeps decoding ----------
    new_mask = torch.tensor([[True], [False]])
    cached_logits = model(
        torch.tensor([[8], [0]]),
        token_mask=new_mask,
        cache=cache,
    )

    expected = model(torch.tensor([[1, 2, 3, 6, 8]]))[:, -1:, :]
    torch.testing.assert_close(
        cached_logits[0:1],
        expected,
        rtol=1e-4,
        atol=1e-5,
    )
    assert cache.lengths.tolist() == [5, 3]
    assert torch.isfinite(cached_logits).all()


@pytest.mark.parametrize("shape", [(0,), (1, 0), (1, 2, 3)])
def test_rejects_unsupported_input_shapes(shape):
    """Rejects empty dimensions and tensors with the wrong rank."""
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


def test_causal_attention_masks_padding_and_empty_rows():
    """Checks causal visibility, padding exclusion and fully masked queries."""
    # ---------- 1. use equal scores so visible values are averaged ----------
    q = torch.zeros(2, 1, 3, 1)
    k = torch.zeros(2, 1, 3, 1)
    # q/k: [B=2, H=1, T=3, D=1]

    v = torch.tensor(
        [
            [[[2.0], [6.0], [100.0]]],
            [[[10.0], [20.0], [30.0]]],
        ]
    )
    # v: [2, 1, 3, 1]
    # The first request's final value is padding and must not contribute.

    positions = torch.tensor([[0, 1, 2], [0, 1, 2]])
    key_mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
        ]
    )
    query_mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
        ]
    )

    # ---------- 2. run attention with padding masks ----------
    output = causal_attention(
        q,
        k,
        v,
        positions,
        key_mask=key_mask,
        query_mask=query_mask,
    )

    # ---------- 3. check causal averages and zero outputs ----------
    expected = torch.tensor(
        [
            [[[2.0], [4.0], [0.0]]],
            [[[0.0], [0.0], [0.0]]],
        ]
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected)

    # ---------- 4. let a valid query look past a padded key ----------
    query_mask[0, 2] = True
    output = causal_attention(
        q,
        k,
        v,
        positions,
        key_mask=key_mask,
        query_mask=query_mask,
    )

    # Only values 2 and 6 are visible: the padded value 100 is excluded.
    torch.testing.assert_close(output[0, 0, 2, 0], torch.tensor(4.0))


@torch.inference_mode()
def test_right_padded_batch_matches_individual_forward():
    """Checks valid batch logits against separate unpadded requests."""
    # ---------- 1. build a small model and two different-length requests ----------
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(make_config()).eval()

    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 0],
        ]
    )
    token_mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
        ]
    )
    # input_ids/token_mask: [B=2, T=3]

    # ---------- 2. compare each valid prefix with a separate forward pass ----------
    batched = model(input_ids, token_mask=token_mask)
    # batched: [2, 3, V]

    for row, length in enumerate((3, 2)):
        individual = model(input_ids[row : row + 1, :length])
        # individual: [1, length, V]

        torch.testing.assert_close(
            batched[row : row + 1, :length],
            individual,
            rtol=1e-4,
            atol=1e-5,
        )

    # ---------- 3. verify that padding token values do not affect valid logits ----------
    changed_ids = input_ids.clone()
    changed_ids[1, 2] = 9
    changed = model(changed_ids, token_mask=token_mask)

    torch.testing.assert_close(
        batched[token_mask],
        changed[token_mask],
        rtol=1e-4,
        atol=1e-5,
    )
    assert torch.isfinite(batched).all()


@torch.inference_mode()
def test_unpadded_batch_matches_individual_forward():
    """Checks that omitting the mask treats every position as valid."""
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(make_config()).eval()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    batched = model(input_ids)
    individual = torch.cat(
        [model(input_ids[row : row + 1]) for row in range(2)],
        dim=0,
    )

    torch.testing.assert_close(batched, individual, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "token_mask",
    [
        [[False, False, False]],
        [[False, True, True]],
        [[True, False, True]],
    ],
)
def test_forward_rejects_empty_or_non_right_padded_rows(token_mask):
    """Rejects empty requests, left padding and gaps between valid tokens."""
    model = Qwen3ForCausalLM(make_config())
    with pytest.raises(ValueError, match="token_mask"):
        model(
            torch.tensor([[1, 2, 3]]),
            token_mask=torch.tensor(token_mask),
        )


def test_cache_rejects_mismatched_batch_before_writing():
    """Rejects a cache created for a different fixed batch size."""
    model = Qwen3ForCausalLM(make_config())
    cache = model.create_kv_cache(max_seq_len=4, batch_size=1)
    input_ids = torch.tensor([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="batch"):
        model(input_ids, cache=cache)

    assert cache.lengths.tolist() == [0]
