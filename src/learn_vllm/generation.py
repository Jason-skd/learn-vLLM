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


@dataclass(frozen=True)
class GenerationRequest:
    input_ids: Tensor  # [1, L]
    max_new_tokens: int
    eos_token_ids: int | list[int] | None = None


def _normalize_eos_token_ids(
    eos_token_ids: int | list[int] | None,
) -> set[int]:
    """Normalizes optional scalar or list EOS IDs for fast membership checks."""
    if eos_token_ids is None:
        return set()
    return set(eos_token_ids if isinstance(eos_token_ids, list) else [eos_token_ids])


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
    eos_ids = _normalize_eos_token_ids(eos_token_ids)

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


@torch.inference_mode()
def greedy_generate_batch(
    model: nn.Module,
    requests: list[GenerationRequest],
    *,
    cache: KVCache | None = None,
) -> list[GenerationResult]:
    """Generates a fixed batch with independent EOS and length limits.

    Prompts are right-padded internally for batched model calls. Each returned
    result removes padding and contains only that request's valid positions.
    Finished requests stay in their batch slots but no longer write KV cache,
    produce output tokens, or grow their cache lengths.

    Args:
        model: A causal language model accepting optional token_mask and cache.
        requests: Nonempty requests with [1, L] prompts and independent limits.
        cache: Optional empty cache whose batch_size equals len(requests).
    Returns:
        One GenerationResult per input request, in input order.
    """
    # shape notation: B=batch, L=input tokens, N=generated tokens, V=vocab size

    # ---------- 1. validate requests and optional cache ----------
    if not requests:
        raise ValueError("requests must not be empty.")

    first_input = requests[0].input_ids
    for request in requests:
        if (
            request.input_ids.ndim != 2
            or request.input_ids.shape[0] != 1
            or request.input_ids.shape[1] == 0
        ):
            raise ValueError("Each input_ids must have shape [1, L] with L > 0.")
        if request.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if (
            request.input_ids.dtype != first_input.dtype
            or request.input_ids.device != first_input.device
        ):
            raise ValueError("All input_ids must share dtype and device.")

    batch_size = len(requests)
    prompt_lengths = torch.tensor(
        [request.input_ids.shape[1] for request in requests],
        dtype=torch.int64,
        device=first_input.device,
    )
    max_prompt_length = int(prompt_lengths.max().item())

    if cache is not None:
        if cache.batch_size != batch_size:
            raise ValueError("cache batch size must match requests.")
        if cache.lengths.any().item():
            raise ValueError("cache must be empty before batch prefill.")
        for request, prompt_length in zip(
            requests, prompt_lengths.tolist(), strict=True
        ):
            required = prompt_length + request.max_new_tokens
            if required > cache.max_seq_len:
                raise ValueError(
                    f"KV cache capacity exceeded for one request: need "
                    f"{required}, capacity is {cache.max_seq_len}."
                )

    # ---------- 2. right-pad prompts and run one batched prefill ----------
    padded_input_ids = torch.zeros(
        (batch_size, max_prompt_length),
        dtype=first_input.dtype,
        device=first_input.device,
    )
    prompt_mask = torch.arange(max_prompt_length, device=first_input.device).unsqueeze(
        0
    ) < prompt_lengths.unsqueeze(1)

    contexts = []
    for row, request in enumerate(requests):
        length = request.input_ids.shape[1]
        padded_input_ids[row, :length] = request.input_ids[0]
        contexts.append(request.input_ids.clone())

    prefill_output = model(
        padded_input_ids,
        token_mask=prompt_mask,
        cache=cache,
    )
    # prefill_output: [B, max L, V]

    row_indices = torch.arange(batch_size, device=first_input.device)

    # ---------- 3. prepare independent generation state ----------
    prefill_logits = [
        prefill_output[row : row + 1, :length].float().cpu().clone()
        for row, length in enumerate(prompt_lengths.tolist())
    ]
    generated_ids: list[list[Tensor]] = [[] for _ in requests]
    generation_logits: list[list[Tensor]] = [[] for _ in requests]
    stop_reasons: list[StopReason] = ["max_new_tokens"] * batch_size
    eos_sets = [_normalize_eos_token_ids(request.eos_token_ids) for request in requests]

    finished = torch.zeros(batch_size, dtype=torch.bool, device=first_input.device)
    step_input = padded_input_ids
    query_mask = prompt_mask
    valid_lengths = prompt_lengths.clone()
    max_steps = max(request.max_new_tokens for request in requests)

    # ---------- 4. generate until every request has its own stop reason ----------
    for step in range(max_steps):
        if step == 0:
            output_logits = prefill_output
        else:
            output_logits = model(
                step_input,
                token_mask=query_mask,
                cache=cache,
            )

        if step == 0:
            # Prefill rows have different valid lengths and may contain padding.
            last_logits = prefill_output[row_indices, prompt_lengths - 1, :]
        elif cache is None:
            # Right padding means the next-token position is each row's last
            # valid context position rather than the final tensor column.
            last_indices = (valid_lengths - 1).clamp_min(0)
            last_logits = output_logits[row_indices, last_indices, :]
        else:
            # Cached decode after prefill receives one new token per request.
            last_logits = output_logits[:, -1, :]

        token = last_logits.argmax(dim=-1, keepdim=True)
        # token: [B, 1]

        token_cpu = token[:, 0].cpu()
        logits_cpu = last_logits.float().cpu().clone()
        token_values = token_cpu.tolist()

        for row, (request, eos_ids) in enumerate(zip(requests, eos_sets, strict=True)):
            if finished[row].item():
                continue

            generated_ids[row].append(token_cpu[row : row + 1].unsqueeze(0))
            generation_logits[row].append(logits_cpu[row : row + 1])

            reached_eos = token_values[row] in eos_ids
            reached_limit = len(generated_ids[row]) >= request.max_new_tokens

            if reached_eos:
                # EOS takes precedence when EOS and the length limit coincide.
                stop_reasons[row] = "eos"
                finished[row] = True
            elif reached_limit:
                stop_reasons[row] = "max_new_tokens"
                finished[row] = True

        if finished.all().item():
            break

        if cache is None:
            # Uncached mode rebuilds a right-padded context once per step.
            for row in range(batch_size):
                if not finished[row].item():
                    contexts[row] = torch.cat(
                        (contexts[row], token[row : row + 1]),
                        dim=1,
                    )

            valid_lengths = torch.tensor(
                [context.shape[1] for context in contexts],
                dtype=torch.int64,
                device=first_input.device,
            )
            max_context_length = int(valid_lengths.max().item())
            step_input = torch.zeros(
                (batch_size, max_context_length),
                dtype=first_input.dtype,
                device=first_input.device,
            )
            query_mask = torch.zeros_like(step_input, dtype=torch.bool)

            for row, context in enumerate(contexts):
                length = context.shape[1]
                if not finished[row].item():
                    step_input[row, :length] = context[0]
                    query_mask[row, :length] = True
        else:
            # Finished rows retain placeholders but their mask is all False.
            step_input = token
            query_mask = (~finished).unsqueeze(1)

    # ---------- 5. pack results without padding ----------
    return [
        GenerationResult(
            prefill_logits=prefill_logits[row],
            generated_ids=torch.cat(generated_ids[row], dim=1),
            generation_logits=torch.stack(generation_logits[row], dim=1),
            stop_reason=stop_reasons[row],
        )
        for row in range(batch_size)
    ]
