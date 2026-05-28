"""Planned evaluation helper for Week 3 and later.

Expected input:
    A PyTorch binary classifier, feature tensor, binary labels, and sensitive
    group indicators.

Expected output:
    A dictionary with clean accuracy and hard-label group disparity metrics.

Status:
    Planned implementation scaffold for small offline checks. Full clean and
    robust evaluation, CSV logging, Fairlearn cross-checks, and final reporting
    are deferred to later weeks.
"""

from __future__ import annotations

import torch
from torch import nn

from fair_safe_opt.metrics.classification import accuracy_from_logits
from fair_safe_opt.metrics.fairness import (
    demographic_parity_difference,
    equal_opportunity_difference,
)


def evaluate_binary_classifier(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
) -> dict[str, float]:
    """Preliminary evaluation helper for small offline checks."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return {
        "accuracy": accuracy_from_logits(logits, y),
        "demographic_parity_difference": demographic_parity_difference(preds, sensitive),
        "equal_opportunity_difference": equal_opportunity_difference(y, preds, sensitive),
    }
