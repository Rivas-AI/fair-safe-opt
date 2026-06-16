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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from fair_safe_opt.data.adult import AdultSplit
import matplotlib.pyplot as plt

from fair_safe_opt.training.erm import TrainingHistory, train_erm
from fair_safe_opt.metrics.classification import accuracy_from_logits
from fair_safe_opt.metrics.fairness import (
    demographic_parity_difference,
    equal_opportunity_difference,
    accuracy_by_group,
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


def set_seed(seed: int = 0):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)


def plot_losses(history: TrainingHistory):
    plt.plot(history.train_losses, label="Training Loss")
    plt.plot(history.val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_accuracies(history: TrainingHistory):
    plt.plot(history.train_accuracies, label="Training Accuracy")
    plt.plot(history.val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def print_metrics(preds: torch.tensor, y: torch.tensor, a: torch.tensor):
    """Outputs accuracies, confusion matrix, demographic parity, and equal opportunity"""
    accuracy_total = (preds == y).float().mean().item()
    accuracy_by_group_dict = accuracy_by_group(preds, y, a)

    print(f"\nTotal Test Accuracy: {accuracy_total:.4f}\n")
    print(f"Group 0 Test Accuracy (Men): {accuracy_by_group_dict[0]:.4f}\n")
    print(f"Group 1 Test Accuracy (Women): {accuracy_by_group_dict[1]:.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))

    print("\nDemographic Parity Difference:")
    print(demographic_parity_difference(preds, a))

    print("\nEqual Opportunity Difference:")
    print(equal_opportunity_difference(preds, y, a))


def evaluate_model(
    model: nn.Module, split: AdultSplit, method: str = "erm", show_plots: bool = True
) -> None:
    """Trains the model and outputs training history plots and metrics."""
    if method == "erm":
        history: TrainingHistory = train_erm(
            model,
            torch.tensor(split.x_train),
            torch.tensor(split.y_train),
            torch.tensor(split.x_val),
            torch.tensor(split.y_val),
            epochs=10,
        )
    else:
        ValueError("Enter a valid training method. Valid methods: {'erm'}")

    predictions = model.predict_classes(torch.tensor(split.x_test))
    if show_plots:
        plot_losses(history)
        plot_accuracies(history)
        ConfusionMatrixDisplay.from_predictions(split.y_test, predictions)
        plt.show()

    print_metrics(predictions, torch.tensor(split.y_test), torch.tensor(split.a_test))
