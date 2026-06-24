from __future__ import annotations

import torch
from sklearn.metrics import balanced_accuracy_score


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).to(y.dtype)
    return float((preds == y).float().mean().item())


def balanced_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).to(y.dtype)
    return float(balanced_accuracy_score(y.numpy(), preds.numpy()))
