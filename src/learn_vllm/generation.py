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

    # eos may be None / int / list[int]; normalize to a set for O(1) lookup
    eos_ids = (
        set()
        if eos_token_ids is None
        else set(eos_token_ids if isinstance(eos_token_ids, list) else [eos_token_ids])
    )

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

        token = logits.argmax(dim=-1, keepdim=True)
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
    new_ids = torch.cat(generated, dim=1)
    # new_ids: [1, N]

    return GenerationResult(
        prefill_logits=prefill_logits,
        generated_ids=new_ids,
        generation_logits=torch.stack(step_logits, dim=1),
        stop_reason=stop_reason,
    )
