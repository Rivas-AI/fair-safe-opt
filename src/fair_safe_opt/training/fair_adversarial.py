"""Planned fair-adversarial training helpers for Weeks 6 and 7.

Expected input:
    Binary logits, labels, sensitive-group indicators, selected fairness
    surrogate (`dp` or `eopp`), and a penalty weight.

Expected output:
    Differentiable surrogate or penalized loss tensor for later training-loop
    integration.

Status:
    Planned implementation scaffold. These helpers do not implement the full
    fairness-constrained or fairness-penalized adversarial-training method. Week
    6 and Week 7 must add training-loop tests, configs, logging, and method
    documentation before use in reported experiments.
"""

from __future__ import annotations

import torch
from torch import nn

from fair_safe_opt.metrics.fairness import (
    demographic_parity_surrogate,
    equal_opportunity_surrogate,
)


def fairness_surrogate(
    logits: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    if metric == "dp":
        return demographic_parity_surrogate(probabilities, sensitive)
    if metric == "eopp":
        return equal_opportunity_surrogate(probabilities, y, sensitive)
    raise ValueError(f"Unsupported fairness metric: {metric}")


def fairness_penalized_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    *,
    metric: str = "dp",
    penalty_weight: float = 1.0,
) -> torch.Tensor:
    """Preliminary differentiable fairness-penalty helper.

    This helper is not a complete fairness-constrained adversarial-training
    implementation. Later work must add training-loop tests and documentation
    before using it for reported experiments.
    """
    base_loss = nn.functional.binary_cross_entropy_with_logits(logits, y.float())
    return base_loss + penalty_weight * fairness_surrogate(logits, y, sensitive, metric)
