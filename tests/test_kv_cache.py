import pytest
import torch

from learn_vllm.kv_cache import KVCache


def make_cache(*, max_seq_len: int = 5) -> KVCache:
    """Builds a small CPU cache for shape and position tests."""
    return KVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=3,
        max_seq_len=max_seq_len,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_cache_writes_shared_prefill_range_then_advances():
    """Checks layer-local writes, shared positions and decode appends."""
    # ---------- 1. write the same prefill positions into every layer ----------
    cache = make_cache()

    prefill_key_0 = torch.arange(18.0).reshape(1, 2, 3, 3)
    prefill_value_0 = prefill_key_0 + 100
    prefill_key_1 = prefill_key_0 + 200
    prefill_value_1 = prefill_key_0 + 300
    # prefill tensors: [B=1, Hkv=2, T=3, D=3]

    keys_0, values_0 = cache.append(0, prefill_key_0, prefill_value_0)
    keys_1, values_1 = cache.append(1, prefill_key_1, prefill_value_1)
    # keys/values: [B=1, Hkv=2, Lused=3, D=3]

    assert cache.length == 0
    assert keys_0.shape == keys_1.shape == (1, 2, 3, 3)
    torch.testing.assert_close(keys_0, prefill_key_0, rtol=0, atol=0)
    torch.testing.assert_close(keys_1, prefill_key_1, rtol=0, atol=0)
    torch.testing.assert_close(values_0, prefill_value_0, rtol=0, atol=0)
    torch.testing.assert_close(values_1, prefill_value_1, rtol=0, atol=0)

    # ---------- 2. mark the prefill range as valid ----------
    cache.advance(3)
    assert cache.length == 3

    # ---------- 3. append one decode token after the prefilled range ----------
    decode_key_0 = torch.full((1, 2, 1, 3), 400.0)
    decode_value_0 = torch.full((1, 2, 1, 3), 500.0)
    decode_key_1 = torch.full((1, 2, 1, 3), 600.0)
    decode_value_1 = torch.full((1, 2, 1, 3), 700.0)

    keys_0, values_0 = cache.append(0, decode_key_0, decode_value_0)
    keys_1, values_1 = cache.append(1, decode_key_1, decode_value_1)

    assert keys_0.shape == keys_1.shape == (1, 2, 4, 3)
    torch.testing.assert_close(keys_0[:, :, :3], prefill_key_0, rtol=0, atol=0)
    torch.testing.assert_close(keys_0[:, :, 3:], decode_key_0, rtol=0, atol=0)
    torch.testing.assert_close(keys_1[:, :, :3], prefill_key_1, rtol=0, atol=0)
    torch.testing.assert_close(keys_1[:, :, 3:], decode_key_1, rtol=0, atol=0)
    torch.testing.assert_close(values_0[:, :, 3:], decode_value_0, rtol=0, atol=0)
    torch.testing.assert_close(values_1[:, :, 3:], decode_value_1, rtol=0, atol=0)

    cache.advance(1)
    assert cache.length == 4


def test_cache_rejects_prefill_beyond_capacity_before_writing():
    """Checks that an oversized write fails without advancing shared length."""
    cache = make_cache(max_seq_len=2)
    key = torch.zeros(1, 2, 3, 3)
    value = torch.zeros(1, 2, 3, 3)

    with pytest.raises(ValueError, match="capacity"):
        cache.append(0, key, value)

    assert cache.length == 0
