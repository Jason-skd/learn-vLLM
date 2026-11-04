from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .kv_cache import KVCache

StopReason = Literal["eos", "max_new_tokens"]


@dataclass(frozen=True)
class GenerationResult:
    prefill_logits: Tensor  # [1, L, V]
    generated_ids: Tensor  # [1, N]
    generation_logits: Tensor  # [1, N, V]
    stop_reason: StopReason


@torch.inference_mode()
def greedy_generate(
    model: nn.Module,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: int | list[int] | None = None,
    cache: KVCache | None = None,
) -> GenerationResult:
    """Generates greedy tokens with optional continuous KV cache support.

    Args:
        model: A causal language model returning logits for new input positions.
        input_ids: [1, L], a nonempty prompt without padding.
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_ids: Optional token IDs that stop generation.
        cache: Optional empty KV cache for one request. When omitted, the full
            context is recomputed at every step.
    Returns:
        Packed prefill logits, generated IDs, step logits and stop reason.
    """
    # shape notation: B=1 (batch), L=input tokens, N=generated tokens, V=vocab size
    # In cached mode, later calls pass T=1 new token instead of the full sequence.

    # ---------- 1. validate input ----------
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    # ---------- 2. prepare vars ----------
    step_input = input_ids
    generated, step_logits = [], []

    # eos may be None / int / list[int]; normalize to a set for O(1) lookup
    eos_ids = (
        set()
        if eos_token_ids is None
        else set(eos_token_ids if isinstance(eos_token_ids, list) else [eos_token_ids])
    )

    stop_reason: StopReason = "max_new_tokens"  # default

    # ---------- 3. generate ----------
    for step in range(max_new_tokens):
        output_logits = model(step_input, cache=cache)
        # Without cache: [1, L + step, V]
        # With cache: prefill [1, L, V], then decode [1, 1, V]

        if step == 0:
            # Save the full prefill logits once, as one of the result tensors.
            prefill_logits = output_logits.float().cpu().clone()
            # prefill_logits: [1, L, V]

        logits = output_logits[:, -1, :]
        # logits: [1, V]

        token = logits.argmax(dim=-1, keepdim=True)
        # token: [1, 1]

        step_logits.append(logits.float().cpu().clone())
        generated.append(token.cpu())

        if token.item() in eos_ids:
            stop_reason = "eos"
            break

        if cache is None:
            step_input = torch.cat((step_input, token), dim=1)
            # step_input: [1, L + step + 1]
        else:
            step_input = token
            # step_input: [1, 1]

    # ---------- 4. pack ----------
    new_ids = torch.cat(generated, dim=1)
    # new_ids: [1, N]

    return GenerationResult(
        prefill_logits=prefill_logits,
        generated_ids=new_ids,
        generation_logits=torch.stack(step_logits, dim=1),
        stop_reason=stop_reason,
    )
