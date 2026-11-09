from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

StopReason = Literal["eos", "max_new_tokens"]


@dataclass(frozen=True)
class GenerationResult:
    prefill_logits: Tensor  # [1, L, V]
    generated_ids: Tensor  # [1, N]
    generation_logits: Tensor  # [1, N, V]
    stop_reason: StopReason


def _normalize_eos_ids(eos_token_ids: int | list[int] | None) -> set[int]:
    """Normalizes optional EOS token IDs for constant-time membership checks."""
    if eos_token_ids is None:
        return set()
    return set(eos_token_ids if isinstance(eos_token_ids, list) else [eos_token_ids])


def _select_next_token(logits: Tensor) -> Tensor:
    """Selects the highest-scoring token for each request in the batch."""
    return logits.argmax(dim=-1, keepdim=True)


def _pack_result(
    prefill_logits: Tensor,
    generated: list[Tensor],
    step_logits: list[Tensor],
    stop_reason: StopReason,
) -> GenerationResult:
    """Assembles the already-recorded outputs into the public result object."""
    return GenerationResult(
        prefill_logits=prefill_logits,
        generated_ids=torch.cat(generated, dim=1),
        generation_logits=torch.stack(step_logits, dim=1),
        stop_reason=stop_reason,
    )


@torch.inference_mode()
def greedy_generate(
    model: nn.Module,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: int | list[int] | None = None,
) -> GenerationResult:
    # ---------- 1. validate input ----------
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    # ---------- 2. prepare vars ----------
    sequence = input_ids
    generated, step_logits = [], []
    prefill_logits: Tensor | None = None

    eos_ids = _normalize_eos_ids(eos_token_ids)

    stop_reason: StopReason = "max_new_tokens"  # default

    # ---------- 3. generate ----------
    for step in range(max_new_tokens):
        # At iteration `step`, sequence length is L + step.
        output_logits, _ = model(sequence)
        # output_logits: [1, L + step, V]

        if step == 0:
            # Save the full prefill logits once, as one of the result tensors.
            prefill_logits = output_logits.float().cpu().clone()
            # prefill_logits: [1, L, V]

        logits = output_logits[:, -1, :]
        # logits: [1, V]

        token = _select_next_token(logits)
        # token: [1, 1]

        step_logits.append(logits.float().cpu().clone())
        generated.append(token.cpu())

        if token.item() in eos_ids:
            stop_reason = "eos"
            break

        sequence = torch.cat((sequence, token), dim=1)
        # sequence: [1, L + step + 1]

    # ---------- 4. pack ----------
    # A positive generation limit guarantees at least one forward pass.
    assert prefill_logits is not None
    return _pack_result(
        prefill_logits,
        generated,
        step_logits,
        stop_reason,
    )


@torch.inference_mode()
def greedy_generate_cached(
    model: nn.Module,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: int | list[int] | None = None,
) -> GenerationResult:
    """Greedy-decodes new tokens while reusing each layer's KV cache."""
    # ---------- 1. validate input ----------
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    # ---------- 2. prepare vars ----------
    model_input = input_ids
    past_key_values = None
    generated: list[Tensor] = []
    step_logits: list[Tensor] = []
    prefill_logits: Tensor | None = None
    eos_ids = _normalize_eos_ids(eos_token_ids)
    stop_reason: StopReason = "max_new_tokens"

    # ---------- 3. generate ----------
    for step in range(max_new_tokens):
        # The first call processes the prompt; later calls process one new token.
        output_logits, past_key_values = model(
            model_input,
            past_key_values=past_key_values,
        )
        # output_logits: [1, input_length, V]; cache now covers all seen tokens.

        if step == 0:
            # Save all prompt logits once, matching the uncached result format.
            prefill_logits = output_logits.float().cpu().clone()
            # prefill_logits: [1, prompt_length, V]

        logits = output_logits[:, -1, :]
        # logits: [1, V]

        token = _select_next_token(logits)
        # token: [1, 1]

        step_logits.append(logits.float().cpu().clone())
        generated.append(token.cpu())

        if token.item() in eos_ids:
            stop_reason = "eos"
            break

        # The cache holds prior tokens, so the next call only needs this token.
        model_input = token

    # ---------- 4. pack ----------
    assert prefill_logits is not None
    return _pack_result(
        prefill_logits,
        generated,
        step_logits,
        stop_reason,
    )
