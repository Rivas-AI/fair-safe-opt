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
from pathlib import Path
import shutil
from datetime import datetime
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


def plot_losses(history: TrainingHistory, save_dir: Path):
    plt.plot(history.train_losses, label="Training Loss")
    plt.plot(history.val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(save_dir / "losses.png")
    plt.close()


def plot_accuracies(history: TrainingHistory, save_dir: Path):
    plt.plot(history.train_accuracies, label="Training Accuracy")
    plt.plot(history.val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(save_dir / "accuracies.png")
    plt.close()


def print_metrics(
    preds: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    output_file: Path,
) -> None:
    """Write experiment results as a markdown report."""

    accuracy_total = (preds == y).float().mean().item()
    accuracy_by_group_dict = accuracy_by_group(preds, y, a)

    dp_diff = demographic_parity_difference(preds, a)
    eo_diff = equal_opportunity_difference(preds, y, a)

    with output_file.open("w") as f:
        print("# Experiment Report", file=f)
        print("", file=f)

        print('## Model Specifications', file=f)
        print(' - Architecture: ', file=f)
        print(' - Seed: ', file=f)
        print("", file=f)
        
        print("## Run Summary", file=f)
        print("", file=f)

        print("| Metric | Value |", file=f)
        print("|----------|----------|", file=f)
        print(f"| Total Accuracy | {accuracy_total:.4f} |", file=f)
        print(
            f"| Group 0 Accuracy (Men) | {accuracy_by_group_dict[0]:.4f} |",
            file=f,
        )
        print(
            f"| Group 1 Accuracy (Women) | {accuracy_by_group_dict[1]:.4f} |",
            file=f,
        )
        print(f"| Demographic Parity Difference | {dp_diff:.4f} |", file=f)
        print(f"| Equal Opportunity Difference | {eo_diff:.4f} |", file=f)

        print("", file=f)

        print("## Training Curves", file=f)
        print("", file=f)
        print("![Loss Curves](losses.png)", file=f)
        print("", file=f)
        print("![Accuracy Curves](accuracies.png)", file=f)

        print("", file=f)

        print("## Confusion Matrix", file=f)
        print("", file=f)
        print("![Confusion Matrix](confusion_matrix.png)", file=f)

        print("", file=f)

        print("## Operator Notes", file=f)
        print("", file=f)


def evaluate_model(
    model: nn.Module, split: AdultSplit, seed: int = 0, method: str = "erm", show_plots: bool = True
) -> None:
    """Trains the model and outputs training history plots and metrics."""
    
    root = Path.cwd()
    
    # Create date directory
    datestr = datetime.now().strftime("%Y.%m.%d")
    date_dir = root / 'results' / datestr

    name = f'seed{seed}_' + datetime.now().strftime("%H:%M")
    save_dir = date_dir / name
    
    if Path(save_dir).exists():
        shutil.rmtree(save_dir)
        
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model using specified method
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

    # Print metrics
    predictions = model.predict_classes(torch.tensor(split.x_test))
    
    if show_plots:
        plot_losses(history, save_dir)
        plot_accuracies(history, save_dir)
        ConfusionMatrixDisplay.from_predictions(split.y_test, predictions)
        plt.savefig(save_dir / 'confusion_matrix.png')
        plt.close()
        
    output_file = save_dir / 'metrics.md'
    print_metrics(predictions, torch.tensor(split.y_test), torch.tensor(split.a_test), output_file)
