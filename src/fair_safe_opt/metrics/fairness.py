from __future__ import annotations

import warnings

import torch


def _as_bool_group(sensitive: torch.Tensor, group: int) -> torch.Tensor:
    return sensitive.to(torch.long) == int(group)


def _rate(values: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        warnings.warn(f"No samples available for {name}; returning nan.", RuntimeWarning)
        return torch.tensor(float("nan"), device=values.device)
    return values[mask].float().mean()


def demographic_parity_difference(
    y_pred: torch.Tensor,
    sensitive: torch.Tensor,
) -> float:
    """Hard-label demographic parity difference for two groups."""
    preds = y_pred.float()
    r0 = _rate(preds, _as_bool_group(sensitive, 0), "group 0")
    r1 = _rate(preds, _as_bool_group(sensitive, 1), "group 1")
    return float(torch.abs(r0 - r1).item())


def equal_opportunity_difference(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sensitive: torch.Tensor,
) -> float:
    """Hard-label equal opportunity difference for two groups."""
    positives = y_true.float() == 1
    r0 = _rate(y_pred.float(), positives & _as_bool_group(sensitive, 0), "positive group 0")
    r1 = _rate(y_pred.float(), positives & _as_bool_group(sensitive, 1), "positive group 1")
    return float(torch.abs(r0 - r1).item())


def demographic_parity_surrogate(
    probabilities: torch.Tensor,
    sensitive: torch.Tensor,
) -> torch.Tensor:
    """Squared probability-mean demographic parity surrogate."""
    r0 = _rate(probabilities, _as_bool_group(sensitive, 0), "group 0")
    r1 = _rate(probabilities, _as_bool_group(sensitive, 1), "group 1")
    return (r0 - r1).pow(2)


def equal_opportunity_surrogate(
    probabilities: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: torch.Tensor,
) -> torch.Tensor:
    """Squared probability-mean equal opportunity surrogate."""
    positives = y_true.float() == 1
    r0 = _rate(probabilities, positives & _as_bool_group(sensitive, 0), "positive group 0")
    r1 = _rate(probabilities, positives & _as_bool_group(sensitive, 1), "positive group 1")
    return (r0 - r1).pow(2)
