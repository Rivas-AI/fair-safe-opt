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
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from dataclasses import dataclass

from fair_safe_opt.data.adult import AdultSplit
from fair_safe_opt.training.erm import TrainingHistory
from fair_safe_opt.metrics.classification import accuracy_from_logits
from fair_safe_opt.metrics.fairness import (
    demographic_parity_difference,
    equal_opportunity_difference,
    accuracy_by_group,
    auc_by_group,
    TPR_by_group
)
from fair_safe_opt.attacks.fgsm import fgsm_attack
from fair_safe_opt.attacks.pgd import pgd_attack


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


@dataclass(frozen=True)
class Evaluation:
    total_acc: float
    group0_acc: float
    group1_acc: float

    total_auc: float
    group0_auc: float
    group1_auc: float
    
    group0_advantaged_TPR: float
    group1_advantaged_TPR: float

    dem_parity_diff: float
    equal_opp_diff: float

    hist_plot_name: str
    total_roc_conf_name: str
    group0_roc_conf_name: str
    group1_roc_conf_name: str


def plot_roc_confusion(
    preds: torch.Tensor,
    probs: torch.Tensor,
    y: torch.Tensor,
    out_dir: Path,
    group_name: str,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

    ConfusionMatrixDisplay.from_predictions(y, preds, ax=axes[0])
    RocCurveDisplay.from_predictions(y, probs, ax=axes[1])

    name = f"{group_name}_roc_confusion.png"
    fig.tight_layout()
    fig.savefig(out_dir / name)
    plt.close(fig)

    return name


def plot_history(history: TrainingHistory, out_dir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

    axes[0].plot(history.train_losses, label="Train Loss")
    axes[0].plot(history.val_losses, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.train_accuracies, label="Train Acc")
    axes[1].plot(history.val_accuracies, label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    filename = "training_curves.png"
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)
    return filename


def evaluate_model(
    model: nn.Module,
    split: AdultSplit,
    history: TrainingHistory,
    out_dir: Path,
    threshold: float = 0.5,
    attack_method: str = 'none',
    eps: float | None = None,
    alpha: float | None = None,
    steps: int = 10
) -> Evaluation:
    """Calculates metrics for all groups from a given model and saves graph data."""
    true_labels = torch.tensor(split.y_test)
    sensitive_col = torch.tensor(split.a_test)
    
    # Set features based on attack type.
    if attack_method == 'none':
        features = torch.tensor(split.x_test)
    elif attack_method == 'pgd' or attack_method == 'fgsm':
        print(f"Generating adversarial features. Method: {attack_method}")
        
        if eps is None or eps < 0:
            raise ValueError(f"Incorrect 'eps' parameter value: {eps}")
        
        if attack_method == 'fgsm':
            features = fgsm_attack(model, torch.tensor(split.x_test), true_labels, eps, torch.tensor(split.continuous_mask), torch.tensor(split.feature_min), torch.tensor(split.feature_max))
        elif attack_method == 'pgd':
            if alpha is None or alpha < 0:
                raise ValueError(f"Invalid 'alpha' param value {alpha}")
            features = pgd_attack(model, torch.tensor(split.x_test), true_labels, eps, alpha, steps, torch.tensor(split.continuous_mask), torch.tensor(split.feature_min), torch.tensor(split.feature_max))
    else:
        raise ValueError(f"invalid attack method {attack_method}")
        
        
    # Calculate predictions and probabilities
    with torch.no_grad():
        logits = model(features)

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    # Plot training history
    history_name = plot_history(history, out_dir)

    # Metrics for both groups
    total_accuracy = (preds == true_labels).float().mean().item()
    total_auc_score = roc_auc_score(true_labels.numpy(), probs.numpy())
    demographic_par = demographic_parity_difference(preds, sensitive_col)
    equal_opp = equal_opportunity_difference(true_labels, preds, sensitive_col)
    total_roc_conf_name = plot_roc_confusion(preds, probs, true_labels, out_dir, "combined")

    per_group_acc = accuracy_by_group(preds, true_labels, sensitive_col)
    per_group_auc = auc_by_group(probs, true_labels, sensitive_col)
    per_group_tpr = TPR_by_group(preds, true_labels, sensitive_col)
    
    # Create boolean mask
    group0_mask = sensitive_col == 0
    group0_preds = preds[group0_mask]
    group0_probs = probs[group0_mask]
    group0_labels = true_labels[group0_mask]
    group1_preds = preds[~group0_mask]
    group1_probs = probs[~group0_mask]
    group1_labels = true_labels[~group0_mask]

    # Metrics for group 0
    group0_accuracy = per_group_acc[0]
    group0_auc_score = per_group_auc[0]
    group0_tpr = per_group_tpr[0]
    group0_roc_conf_name = plot_roc_confusion(
        group0_preds, group0_probs, group0_labels, out_dir, "group0"
    )

    # Metrics for group 1
    group1_accuracy = per_group_acc[1]
    group1_auc_score = per_group_auc[1]
    group1_tpr = per_group_tpr[1]
    group1_roc_conf_name = plot_roc_confusion(
        group1_preds, group1_probs, group1_labels, out_dir, "group1"
    )

    evaluation = Evaluation(
        total_acc=total_accuracy,
        group0_acc=group0_accuracy,
        group1_acc=group1_accuracy,
        total_auc=total_auc_score,
        group0_auc=group0_auc_score,
        group1_auc=group1_auc_score,
        group0_advantaged_TPR = group0_tpr,
        group1_advantaged_TPR = group1_tpr,
        dem_parity_diff=demographic_par,
        equal_opp_diff=equal_opp,
        hist_plot_name=history_name,
        total_roc_conf_name=total_roc_conf_name,
        group0_roc_conf_name=group0_roc_conf_name,
        group1_roc_conf_name=group1_roc_conf_name,
    )

    return evaluation
