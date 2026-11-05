import pytest
import torch

from learn_vllm.kv_cache import KVCache


def make_cache(*, batch_size: int, max_seq_len: int = 5) -> KVCache:
    """Builds a small CPU cache for shape and position tests."""
    return KVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=3,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_cache_writes_shared_prefill_range_then_advances():
    """Checks layer-local writes, shared positions and decode appends."""
    # ---------- 1. write the same prefill positions into every layer ----------
    cache = make_cache(batch_size=1)

    prefill_key_0 = torch.arange(18.0).reshape(1, 2, 3, 3)
    prefill_value_0 = prefill_key_0 + 100
    prefill_key_1 = prefill_key_0 + 200
    prefill_value_1 = prefill_key_0 + 300
    token_mask = torch.ones((1, 3), dtype=torch.bool)
    # prefill tensors: [B=1, Hkv=2, T=3, D=3]

    keys_0, values_0 = cache.append(
        0,
        prefill_key_0,
        prefill_value_0,
        token_mask=token_mask,
    )
    keys_1, values_1 = cache.append(
        1,
        prefill_key_1,
        prefill_value_1,
        token_mask=token_mask,
    )
    # keys/values: [B=1, Hkv=2, Lused=3, D=3]

    assert cache.lengths.tolist() == [0]
    assert keys_0.shape == keys_1.shape == (1, 2, 3, 3)
    torch.testing.assert_close(keys_0, prefill_key_0, rtol=0, atol=0)
    torch.testing.assert_close(keys_1, prefill_key_1, rtol=0, atol=0)
    torch.testing.assert_close(values_0, prefill_value_0, rtol=0, atol=0)
    torch.testing.assert_close(values_1, prefill_value_1, rtol=0, atol=0)

    # ---------- 2. mark the prefill range as valid ----------
    cache.advance(token_mask)
    assert cache.lengths.tolist() == [3]

    # ---------- 3. append one decode token after the prefilled range ----------
    decode_key_0 = torch.full((1, 2, 1, 3), 400.0)
    decode_value_0 = torch.full((1, 2, 1, 3), 500.0)
    decode_key_1 = torch.full((1, 2, 1, 3), 600.0)
    decode_value_1 = torch.full((1, 2, 1, 3), 700.0)
    decode_mask = torch.ones((1, 1), dtype=torch.bool)

    keys_0, values_0 = cache.append(
        0,
        decode_key_0,
        decode_value_0,
        token_mask=decode_mask,
    )
    keys_1, values_1 = cache.append(
        1,
        decode_key_1,
        decode_value_1,
        token_mask=decode_mask,
    )

    assert keys_0.shape == keys_1.shape == (1, 2, 4, 3)
    torch.testing.assert_close(keys_0[:, :, :3], prefill_key_0, rtol=0, atol=0)
    torch.testing.assert_close(keys_0[:, :, 3:], decode_key_0, rtol=0, atol=0)
    torch.testing.assert_close(keys_1[:, :, :3], prefill_key_1, rtol=0, atol=0)
    torch.testing.assert_close(keys_1[:, :, 3:], decode_key_1, rtol=0, atol=0)
    torch.testing.assert_close(values_0[:, :, 3:], decode_value_0, rtol=0, atol=0)
    torch.testing.assert_close(values_1[:, :, 3:], decode_value_1, rtol=0, atol=0)

    cache.advance(decode_mask)
    assert cache.lengths.tolist() == [4]


def test_cache_rejects_prefill_beyond_capacity_before_writing():
    """Checks that an oversized write fails without advancing shared length."""
    cache = make_cache(batch_size=1, max_seq_len=2)
    key = torch.zeros(1, 2, 3, 3)
    value = torch.zeros(1, 2, 3, 3)
    token_mask = torch.ones((1, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="capacity"):
        cache.append(0, key, value, token_mask=token_mask)

    assert cache.lengths.tolist() == [0]


def test_batch_cache_writes_independent_prefixes_and_lengths():
    """Checks variable prompt lengths, padding exclusion and per-request decode."""
    # ---------- 1. prefill two requests with different valid lengths ----------
    cache = make_cache(batch_size=2)
    prefill_key = torch.arange(36.0).reshape(2, 2, 3, 3)
    prefill_value = prefill_key + 100
    token_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )

    keys, values = cache.append(
        0,
        prefill_key,
        prefill_value,
        token_mask=token_mask,
    )
    projected_mask = cache.projected_key_mask(token_mask)

    # ---------- 2. verify compact prefixes and projected visibility ----------
    assert cache.lengths.tolist() == [0, 0]
    assert keys.shape == values.shape == (2, 2, 2, 3)
    torch.testing.assert_close(keys[0, :, :2], prefill_key[0, :, :2])
    torch.testing.assert_close(keys[1, :, :1], prefill_key[1, :, :1])
    torch.testing.assert_close(
        projected_mask,
        torch.tensor([[True, True], [True, False]]),
    )
    torch.testing.assert_close(values[0, :, :2], prefill_value[0, :, :2])

    cache.advance(token_mask)
    assert cache.lengths.tolist() == [2, 1]

    # ---------- 3. decode request zero and finish request one ----------
    decode_key = torch.full((2, 2, 1, 3), 500.0)
    decode_value = torch.full((2, 2, 1, 3), 600.0)
    decode_mask = torch.tensor([[True], [False]])

    keys, values = cache.append(
        0,
        decode_key,
        decode_value,
        token_mask=decode_mask,
    )

    assert keys.shape == values.shape == (2, 2, 3, 3)
    torch.testing.assert_close(keys[0, :, 2:3], decode_key[0])
    torch.testing.assert_close(keys[1, :, 1:2], torch.zeros(2, 1, 3))
    torch.testing.assert_close(
        cache.projected_key_mask(decode_mask),
        torch.tensor(
            [
                [True, True, True],
                [True, False, False],
            ]
        ),
    )

    cache.advance(decode_mask)
    assert cache.lengths.tolist() == [3, 1]
