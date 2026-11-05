from .generation import (
    GenerationRequest,
    GenerationResult,
    greedy_generate,
    greedy_generate_batch,
)

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "greedy_generate",
    "greedy_generate_batch",
]


def main() -> None:
    print("Hello from learn-vllm!")
