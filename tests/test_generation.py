import pytest
import torch
from torch import Tensor, nn
from transformers import Qwen3Config

from learn_vllm.generation import greedy_generate, greedy_generate_cached
from learn_vllm.model import Qwen3ForCausalLM


class LastTokenModel(nn.Module):
    """Returns scripted logits and a simple marker as its cache."""

    def __init__(self, next_tokens: list[int], vocab_size: int = 8):
        super().__init__()
        self.next_tokens = next_tokens
        self.vocab_size = vocab_size
        self.input_lengths: list[int] = []
        self.received_caches: list[int | None] = []

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: int | None = None,
    ) -> tuple[Tensor, int]:
        self.input_lengths.append(input_ids.shape[1])
        self.received_caches.append(past_key_values)

        call_number = len(self.input_lengths)
        token = self.next_tokens[call_number - 1]

        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -100.0,
            dtype=torch.float32,
        )
        logits[:, -1, token] = float(call_number)

        # The marker makes cache handoff observable in tests.
        return logits, call_number


def test_greedy_generate_recomputes_full_context_until_length_limit():
    """Checks the uncached loop, greedy result and packed tensor shapes."""
    model = LastTokenModel([2, 4, 3])
    input_ids = torch.tensor([[1, 6]])

    result = greedy_generate(model, input_ids, max_new_tokens=3)

    assert model.input_lengths == [2, 3, 4]
    assert model.received_caches == [None, None, None]
    torch.testing.assert_close(
        result.generated_ids, torch.tensor([[2, 4, 3]]), rtol=0, atol=0
    )
    assert result.prefill_logits.shape == (1, 2, model.vocab_size)
    assert result.generation_logits.shape == (1, 3, model.vocab_size)
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
    assert model.received_caches == [None, None]
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


def test_greedy_generate_cached_processes_only_new_tokens():
    """Checks one-token decode inputs and cache handoff between model calls."""
    model = LastTokenModel([2, 4, 3])

    result = greedy_generate_cached(
        model,
        torch.tensor([[1, 6]]),
        max_new_tokens=3,
    )

    assert model.input_lengths == [2, 1, 1]
    assert model.received_caches == [None, 1, 2]
    torch.testing.assert_close(
        result.generated_ids, torch.tensor([[2, 4, 3]]), rtol=0, atol=0
    )
    assert result.prefill_logits.shape == (1, 2, model.vocab_size)
    assert result.generation_logits.shape == (1, 3, model.vocab_size)
    assert result.stop_reason == "max_new_tokens"


@pytest.mark.parametrize("eos_token_ids", [4, [4]])
def test_greedy_generate_cached_stops_after_eos(eos_token_ids):
    """Keeps EOS in the output and does not pass it back into the model."""
    model = LastTokenModel([2, 4, 3])

    result = greedy_generate_cached(
        model,
        torch.tensor([[1, 6]]),
        max_new_tokens=5,
        eos_token_ids=eos_token_ids,
    )

    assert model.input_lengths == [2, 1]
    assert model.received_caches == [None, 1]
    torch.testing.assert_close(
        result.generated_ids, torch.tensor([[2, 4]]), rtol=0, atol=0
    )
    assert result.stop_reason == "eos"


def test_greedy_generate_cached_rejects_nonpositive_limit():
    with pytest.raises(ValueError, match="max_new_tokens"):
        greedy_generate_cached(
            LastTokenModel([2]),
            torch.tensor([[1, 6]]),
            max_new_tokens=0,
        )


@torch.inference_mode()
def test_greedy_generate_cached_matches_full_context_model():
    """Compares cached decoding with the full-context baseline on a tiny model."""
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 6, 3]])

    baseline = greedy_generate(model, input_ids, max_new_tokens=4)
    cached = greedy_generate_cached(model, input_ids, max_new_tokens=4)

    torch.testing.assert_close(
        cached.prefill_logits, baseline.prefill_logits, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        cached.generation_logits, baseline.generation_logits, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        cached.generated_ids, baseline.generated_ids, rtol=0, atol=0
    )
    assert cached.stop_reason == baseline.stop_reason
