from __future__ import annotations

import torch


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).to(y.dtype)
    return float((preds == y).float().mean().item())
