import torch


def _flip_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(mask, dtype=torch.float).masked_fill(
        ~mask, value=float("-inf")
    )
