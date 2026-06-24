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

    # Metrics for group 0
    group0_accuracy = per_group_acc[0]
    group0_auc_score = per_group_auc[0]
    group0_tpr = per_group_tpr[0]

    # Metrics for group 1
    group1_accuracy = per_group_acc[1]
    group1_auc_score = per_group_auc[1]
    group1_tpr = per_group_tpr[1]

    return pd.DataFrame({
        "group_0": {
            "balanced_accuracy": group0_accuracy,
            "auc": group0_auc_score,
            "tpr": group0_tpr,
            "demographic_parity_difference": None,
            "equal_opportunity_difference": None,
        },
        "group_1": {
            "balanced_accuracy": group1_accuracy,
            "auc": group1_auc_score,
            "tpr": group1_tpr,
            "demographic_parity_difference": None,
            "equal_opportunity_difference": None,
        },
        "combined": {
            "balanced_accuracy": total_accuracy,
            "auc": total_auc_score,
            "tpr": None,
            "demographic_parity_difference": demographic_par,
            "equal_opportunity_difference": equal_opp,
        },
    })

def save_evaluation_plots(preds, probs, labels, sensitive, history, out_dir) -> None:
    plot_history(history, out_dir)
    plot_roc_confusion(preds, probs, labels, out_dir, "combined")
    group0 = sensitive == 0
    plot_roc_confusion(preds[group0], probs[group0], labels[group0], out_dir, "group0")
    plot_roc_confusion(preds[~group0], probs[~group0], labels[~group0], out_dir, "group1")
