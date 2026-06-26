import torch
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

from fair_safe_opt.training.erm import TrainingHistory

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
    axes[1].set_title("Balanced Accuracy")
    axes[1].legend()

    filename = "training_curves.png"
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)
    return filename