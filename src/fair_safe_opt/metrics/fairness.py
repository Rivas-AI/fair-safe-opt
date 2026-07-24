from __future__ import annotations

import warnings
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import torch


def _as_bool_group(sensitive: torch.Tensor, group: int) -> torch.Tensor:
    """Return a boolean mask for the specified group in the sensitive attribute.

    Args:
        sensitive: a tensor representing the value of the sensitive attribute for all samples.
        group: the group to create a mask for (0 or 1).

    Returns:
        A boolean mask for the specified group. True for all samples in the group, False otherwise.
    """
    return sensitive.to(torch.long) == int(group)


def _rate(values: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    """Compute the percentage of positive predictions within a group.

    Args:
        values (torch.Tensor): Tensor of values per sample.
        mask (torch.Tensor): A boolean mask indicating which values are in the group.
        name (str): The name of the group (for warning messages).

    Returns:
        torch.Tensor: the percentage of positive predictions in the group
    """
    if int(mask.sum().item()) == 0:
        warnings.warn(f"No samples available for {name}; returning nan.", RuntimeWarning)
        return torch.tensor(float("nan"), device=values.device)
    return values[mask].float().mean()


def demographic_parity_difference(
    y_pred: torch.Tensor,
    sensitive: torch.Tensor,
) -> float:
    """Hard-label demographic parity difference for two groups.
    
    POSITIVE VALUES INDICATE THAT GO HAS THE HIGHER RATE. 

    Args:
        y_pred: a tensor representing the predicted labels for all samples.
        sensitive: a tensor representing the value of the sensitive attribute for all samples.

    Returns:
        float: The demographic parity difference between the two groups.
    """
    preds = y_pred.float()
    r0 = _rate(preds, _as_bool_group(sensitive, 0), "group 0")
    r1 = _rate(preds, _as_bool_group(sensitive, 1), "group 1")

    if torch.isnan(r0) or torch.isnan(r1):
        return torch.nan

    return float(r0 - r1)


def equal_opportunity_difference(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sensitive: torch.Tensor,
) -> float:
    """Calculated true positive rates for two groups and calculated their difference

    Args:
        y_true: a tensor representing the true labels for all samples.
        y_pred: a tensor representing the predicted labels for all samples.
        sensitive: a tensor representing the value of the sensitive attribute for all samples.

    Returns:
        float: The equal opportunity difference between the two groups.
    """
    positives = y_true.float() == 1
    r0 = _rate(y_pred.float(), positives & _as_bool_group(sensitive, 0), "positive group 0")
    r1 = _rate(y_pred.float(), positives & _as_bool_group(sensitive, 1), "positive group 1")
    return float(r0 - r1)


def demographic_parity_surrogate(
    probabilities: torch.Tensor,
    sensitive: torch.Tensor,
) -> torch.Tensor:
    """Squared probability-mean demographic parity surrogate. Smooth function used for training.

    Definition:
        (E[p(x) | A = 0] - E[p(x) | A = 1]) ** 2
    """
    r0 = _rate(probabilities, _as_bool_group(sensitive, 0), "group 0")
    r1 = _rate(probabilities, _as_bool_group(sensitive, 1), "group 1")
    return (r0 - r1).pow(2)


def equal_opportunity_surrogate(
    probabilities: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: torch.Tensor,
) -> torch.Tensor:
    """Squared probability-mean equal opportunity surrogate. Smooth function used for training.

    Definition:
        (E[p(x) | A = 0, Y = 1] - E[p(x) | A = 1, Y = 1]) ** 2
    """
    positives = y_true.float() == 1
    r0 = _rate(probabilities, positives & _as_bool_group(sensitive, 0), "positive group 0")
    r1 = _rate(probabilities, positives & _as_bool_group(sensitive, 1), "positive group 1")
    return (r0 - r1).pow(2)


def auc_by_group(
    probs: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: torch.Tensor,
) -> dict[int, float]:
    """AUC separately for sensitive groups 0 and 1."""
    result = {}
    for group in (0, 1):
        mask = _as_bool_group(sensitive, group)
        if mask.sum().item() == 0:
            warnings.warn(f"No samples for group {group}; returning nan.", RuntimeWarning)
            result[group] = float("nan")
        else:
            result[group] = float(roc_auc_score(y_true[mask].numpy(), probs[mask].numpy()))
    return result
    
def balanced_accuracy_by_group(
    predictions: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: torch.Tensor,
) -> dict[int, float]:
    """Balanced accuracy (average of TPR and TNR) separately for sensitive groups 0 and 1."""
    result = {}
    for group in (0, 1):
        mask = _as_bool_group(sensitive, group)
        if mask.sum().item() == 0:
            warnings.warn(f"No samples for group {group}; returning nan.", RuntimeWarning)
            result[group] = float("nan")
        else:
            result[group] = float(balanced_accuracy_score(y_true[mask].numpy(), predictions[mask].numpy()))
    return result


def TPR_by_group(
    predictions: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: torch.Tensor,
) -> dict[int, float]:
    """True positive rate separately for sensitive groups 0 and 1."""
    positives = y_true.to(torch.long) == 1

    tpr_combined = _rate(predictions.float(), positives, "combined positive group")
    tpr0 = _rate(predictions.float(), positives & _as_bool_group(sensitive, 0), "positive group 0")
    tpr1 = _rate(predictions.float(), positives & _as_bool_group(sensitive, 1), "positive group 1")

    return {
        0: float(tpr0.item()),
        1: float(tpr1.item()),
        2: float(tpr_combined.item())
    }
