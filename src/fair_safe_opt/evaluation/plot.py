import torch
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd

from fair_safe_opt.training.erm import TrainingHistory
from fair_safe_opt.reporting.robustness_tables import seed_averages_and_stddev

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

def line_plot(
    x: pd.Series,
    y: pd.Series,
    ax: plt.Axes,
    stddev: pd.Series | None = None,
    label: str | None = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
) -> plt.Axes:
    """Draw a line of y vs x with an optional ±1 std band onto the given ax."""
    ax.plot(x, y, marker="o", label=label)

    if stddev is not None:
        ax.fill_between(x, y - stddev, y + stddev, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return ax


def metric_vs_eps_plot(
    dfs: dict[str, pd.DataFrame],
    metric: str,
    axes,
) -> None:
    '''Overlay one line per regime, one panel per architecture, onto `axes`.

    dfs: {label -> averaged df with stddevs}, e.g. {"Baseline": base, "Madry": madry}.
    axes: 1D array of Axes, one per architecture, from the caller.
    '''
    # architectures taken from the first df; assumes all dfs share them
    architectures = next(iter(dfs.values()))['architecture'].unique()

    for ax, architecture in zip(axes, architectures):
        for label, df in dfs.items():
            sub = df[df['architecture'] == architecture].sort_values('epsilon')
            line_plot(
                sub['epsilon'],
                sub[metric],
                ax,
                stddev=sub[f'{metric}_std'],
                label=label,
                title=architecture,
                xlabel="ε",
                ylabel=metric,
            )
        ax.legend()


