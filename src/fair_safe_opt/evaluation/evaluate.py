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
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from typing import TextIO

from fair_safe_opt.utils.seeds import set_seed
from fair_safe_opt.models.logistic import LogisticRegression
from fair_safe_opt.models.mlp import SmallMLP
from fair_safe_opt.data.adult import AdultSplit, load_adult
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


def create_dir(architecture, method, seed):
    '''Create a out_directory for today, and one inside for this run'''
    datestr = datetime.now().strftime("%Y.%m.%d")
    timestr = datetime.now().strftime("%H%M")
    date_dir = Path.cwd() / 'results' / datestr

    name = f'{architecture}_seed{seed}_{method}_{timestr}'
    out_dir = date_dir / name
    
    if Path(out_dir).exists():
        shutil.rmtree(out_dir)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    return out_dir


def plot_history(history: TrainingHistory, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

    axes[0].plot(history.train_losses, label="Train Loss")
    axes[0].plot(history.val_losses, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.train_accuracies, label="Train Acc")
    axes[1].plot(history.val_accuracies, label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png")
    plt.close(fig)


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


def print_combined_metrics(
    preds: torch.Tensor,
    probs: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    f: TextIO,
    out_dir: Path
) -> None:
    """Write combined group results in markdown report."""

    accuracy_total = (preds == y).float().mean().item()
    auc = roc_auc_score(y, probs)

    dp_diff = demographic_parity_difference(preds, a)
    eo_diff = equal_opportunity_difference(y, preds, a)

    roc_conf = plot_roc_confusion(preds, probs, y, out_dir, 'combined')
    
    print("| Metric | Value |", file=f)
    print("|----------|----------|", file=f)
    print(f"| Accuracy | {accuracy_total:.4f} |", file=f)
    print(f"| ROC AUC | {auc:.4f}", file=f)
    print(f"| Demographic Parity Difference | {dp_diff:.4f} |", file=f)
    print(f"| Equal Opportunity Difference | {eo_diff:.4f} |", file=f)
    print("", file=f)

    print("## Confusion Matrix & ROC Curve", file=f)
    print("", file=f)
    print(f"![Confusion Matrix]({roc_conf})", file=f)


def print_individual_metrics(
    preds: torch.Tensor,
    probs: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    out_dir: Path,
    f: TextIO,
    name: str
) -> None:
    """Write individual group results in markdown report."""
    preds = preds[mask]
    probs = probs[mask]
    y = y[mask]
    
    accuracy = (preds == y).float().mean().item()
    auc = roc_auc_score(y, probs)
    
    plot = plot_roc_confusion(preds, probs, y, out_dir, name)
    
    print("| Metric | Value |", file=f)
    print("|----------|----------|", file=f)
    print(f"| Accuracy | {accuracy:.4f} |", file=f)
    print(f"| ROC AUC | {auc:.4f}", file=f)
    print("", file=f)

    print("### Confusion Matrix & ROC Curve", file=f)
    print("", file=f)
    print(f"![confusion_roc]({plot})", file=f)
    print('', file=f)

def evaluate_model(
    architecture: str = 'logistic', 
    method: str = "erm", 
    seed: int = 0, 
    epochs: int = 10, 
    threshold: float = 0.5
) -> None:
    """Trains the model and outputs training history plots and metrics."""
    set_seed(seed)
    out_dir = create_dir(architecture, method, seed)
    
    split: AdultSplit = load_adult()
    
    if architecture == 'logistic':
        model = LogisticRegression(split.x_train.shape[1])
    elif architecture == 'small_mlp':
        model = SmallMLP(split.x_train.shape[1])
    else:
        raise ValueError("architecture not in {'logistic', 'small_mlp'}")
    
    # Train the model using specified method
    if method == "erm":
        history: TrainingHistory = train_erm(
            model,
            torch.tensor(split.x_train),
            torch.tensor(split.y_train),
            torch.tensor(split.x_val),
            torch.tensor(split.y_val),
            epochs=epochs,
        )
    else:
        raise ValueError("method not in {'erm'}")

    # Create report
    with torch.no_grad():
        logits = model(torch.tensor(split.x_test))
        
    
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    
    # Create graphs
    plot_history(history, out_dir)
    
    output_file = out_dir / 'metrics.md'
    with open(output_file, 'w') as f:
        print("# Full Evaluation", file=f)
        print('', file=f)
        print('## Model Specifications', file=f)
        print(f' - Architecture: {architecture}', file=f)
        print(f' - Seed: {seed}', file=f)
        print(f' - Method: {method}', file=f)
        print("", file=f)
        
        print('## Training History', file=f)
        print("", file=f)
        print("![Training Curves](training_curves.png)", file=f)
        print('', file=f)
        
        print('## Metrics', file=f)
        print('', file=f)
        
        y = torch.tensor(split.y_test)
        a = torch.tensor(split.a_test)
        mask = torch.tensor(split.a_test == 0)
        
        print("### Combined", file=f)
        print('', file=f)
        print_combined_metrics(preds, probs, y, a, f, out_dir)
        
        print("### Group 0 (Men)", file=f)
        print('', file=f)
        
        print_individual_metrics(preds, probs, y, mask, out_dir, f, name='group0')
        
        print("### Group 1 (Women)", file=f)
        print('', file=f)
        
        print_individual_metrics(preds, probs, y, ~mask, out_dir, f, name='group1')
