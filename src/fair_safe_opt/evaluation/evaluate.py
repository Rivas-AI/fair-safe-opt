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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import pandas as pd

from fair_safe_opt.metrics.classification import balanced_accuracy_from_logits
from fair_safe_opt.metrics.fairness import (
    demographic_parity_difference,
    equal_opportunity_difference,
    balanced_accuracy_by_group,
    auc_by_group,
    TPR_by_group
)
from fair_safe_opt.evaluation.plot import plot_history, plot_roc_confusion


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
        "balanced_accuracy": balanced_accuracy_from_logits(logits, y),
        "demographic_parity_difference": demographic_parity_difference(preds, sensitive),
        "equal_opportunity_difference": equal_opportunity_difference(y, preds, sensitive),
    }


def create_metric_table(preds: torch.Tensor, probs: torch.Tensor, labels: torch.Tensor, sensitive: torch.Tensor) -> pd.DataFrame:
    """Calculates metrics for all groups from a given model."""
    total_accuracy = balanced_accuracy_score(labels.numpy(), preds.numpy())
    total_auc_score = roc_auc_score(labels.numpy(), probs.numpy())
    demographic_par = demographic_parity_difference(preds, sensitive)
    equal_opp = equal_opportunity_difference(labels, preds, sensitive)
    
    per_group_acc = balanced_accuracy_by_group(preds, labels, sensitive)
    per_group_auc = auc_by_group(probs, labels, sensitive)
    per_group_tpr = TPR_by_group(preds, labels, sensitive)
    combined_tpr = per_group_tpr[2]  # True positive rate for the combined groups
    # Metrics for group 0
    group0_accuracy = per_group_acc[0]
    group0_auc_score = per_group_auc[0]
    group0_tpr = per_group_tpr[0]

    # Metrics for group 1
    group1_accuracy = per_group_acc[1]
    group1_auc_score = per_group_auc[1]
    group1_tpr = per_group_tpr[1]

    df = pd.DataFrame({
        "group": {
            "0": "group_0",
            "1": "group_1",
            "2": "combined"
        },
        "balanced_accuracy": {
            "0": group0_accuracy,
            "1": group1_accuracy,
            "2": total_accuracy,
        },
        "auc": {
            "0": group0_auc_score,
            "1": group1_auc_score,
            "2": total_auc_score,
        },
        "advantaged_tpr": {
            "0": group0_tpr,
            "1": group1_tpr,
            "2": combined_tpr,
        },
        "demographic_parity_difference": {
            "0": None,
            "1": None,
            "2": demographic_par,
        },
        "equal_opportunity_difference": {
            "0": None,
            "1": None,
            "2": equal_opp,
        },
    })
    
    return df

def save_evaluation_plots(preds, probs, labels, sensitive, out_dir, prefix='') -> None:
    '''Plot confusion matrices and roc graphs for each group'''
    plot_roc_confusion(preds, probs, labels, out_dir, f"{prefix}_combined")
    group0 = sensitive == 0
    plot_roc_confusion(preds[group0], probs[group0], labels[group0], out_dir, f"{prefix}_group0")
    plot_roc_confusion(preds[~group0], probs[~group0], labels[~group0], out_dir, f"{prefix}_group1")
