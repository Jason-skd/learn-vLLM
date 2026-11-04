import pytest
import torch
from torch import Tensor, nn

from learn_vllm.generation import greedy_generate
from learn_vllm.kv_cache import KVCache


class LastTokenModel(nn.Module):
    """Returns scripted next-token logits while recording each input length."""

    def __init__(self, next_tokens: list[int], vocab_size: int = 8):
        super().__init__()
        self.next_tokens = next_tokens
        self.vocab_size = vocab_size
        self.input_lengths: list[int] = []
        self.cache_inputs: list[KVCache | None] = []

    def forward(self, input_ids: Tensor, *, cache: KVCache | None = None) -> Tensor:
        self.input_lengths.append(input_ids.shape[1])
        self.cache_inputs.append(cache)
        if cache is not None:
            cache.advance(input_ids.shape[1])

        token = self.next_tokens[len(self.input_lengths) - 1]
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -100.0,
            dtype=torch.float32,
        )
        logits[:, -1, token] = float(len(self.input_lengths))
        return logits


def test_greedy_generate_recomputes_full_context_until_length_limit():
    """Checks the uncached loop, greedy result and packed tensor shapes."""
    model = LastTokenModel([2, 4, 3])
    input_ids = torch.tensor([[1, 6]])

    result = greedy_generate(model, input_ids, max_new_tokens=3)

    assert model.input_lengths == [2, 3, 4]
    torch.testing.assert_close(
        result.generated_ids, torch.tensor([[2, 4, 3]]), rtol=0, atol=0
    )
    assert result.prefill_logits.shape == (1, 2, model.vocab_size)
    assert result.generation_logits.shape == (1, 3, model.vocab_size)
    assert result.stop_reason == "max_new_tokens"
    torch.testing.assert_close(input_ids, torch.tensor([[1, 6]]), rtol=0, atol=0)


def test_greedy_generate_uses_cache_and_feeds_one_token_at_a_time():
    """Checks the cached call pattern, greedy result and packed tensor shapes."""
    model = LastTokenModel([2, 4, 3])
    cache = KVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        max_seq_len=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    input_ids = torch.tensor([[1, 6]])

    result = greedy_generate(
        model,
        input_ids,
        max_new_tokens=2,
        cache=cache,
    )

    assert model.input_lengths == [2, 1]
    assert all(actual is cache for actual in model.cache_inputs)
    assert cache.length == 3
    torch.testing.assert_close(
        result.generated_ids, torch.tensor([[2, 4]]), rtol=0, atol=0
    )
    assert result.prefill_logits.shape == (1, 2, model.vocab_size)
    assert result.generation_logits.shape == (1, 2, model.vocab_size)
    assert result.stop_reason == "max_new_tokens"
    torch.testing.assert_close(input_ids, torch.tensor([[1, 6]]), rtol=0, atol=0)


@pytest.mark.parametrize("eos_token_ids", [4, [4]])
def test_greedy_generate_records_eos_and_stops_before_next_forward(eos_token_ids):
    """Keeps EOS in the output and stops without another full-context pass."""
    model = LastTokenModel([2, 4, 3])

    result = greedy_generate(
        model,
        torch.tensor([[1, 6]]),
        max_new_tokens=5,
        eos_token_ids=eos_token_ids,
    )

    assert model.input_lengths == [2, 3]
    torch.testing.assert_close(
        result.generated_ids, torch.tensor([[2, 4]]), rtol=0, atol=0
    )
    assert result.generation_logits.shape == (1, 2, model.vocab_size)
    assert result.stop_reason == "eos"


def test_greedy_generate_rejects_nonpositive_limit():
    with pytest.raises(ValueError, match="max_new_tokens"):
        greedy_generate(
            LastTokenModel([2]),
            torch.tensor([[1, 6]]),
            max_new_tokens=0,
        )
