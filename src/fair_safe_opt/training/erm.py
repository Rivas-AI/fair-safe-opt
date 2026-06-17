from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import copy
from pathlib import Path

from fair_safe_opt.metrics.classification import accuracy_from_logits


@dataclass(frozen=True)
class TrainingHistory:
    train_losses: list[float]
    train_accuracies: list[float]
    val_losses: list[float]
    val_accuracies: list[float]


def set_device():
    """
    Sets device used for training. Heirarchy goes CUDA, MPS, then CPU if none are available.
    returns:
    - (torch.device): device to be used.
    """
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def train_erm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    save_dir: Path | None = None,
    *,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-2,
    patience: int = 5,
    show_progress: bool = True,
) -> TrainingHistory:
    # Set device
    device = set_device()
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    # Define training variables
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    train_losses: list[float] = []
    train_accuracies: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    epochs_without_improvement = 0
    best_val_loss = float("inf")  # initialize to infinity

    # Code to show progress bar
    if show_progress:
        from tqdm.auto import tqdm

        progress = tqdm(range(epochs), desc="ERM training", unit="epoch")
    else:
        progress = range(epochs)

    # Epoch loop
    for _ in progress:
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb.float())
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            logits_train = model(x)
            logits_val = model(x_val)

        # Append training metrics
        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_accuracy = accuracy_from_logits(logits_train, y)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Calculate Validation loss & acc
        val_loss = nn.functional.binary_cross_entropy_with_logits(logits_val, y_val.float()).item()
        val_accuracy = accuracy_from_logits(logits_val, y_val)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Update Progress Bar
        if show_progress:
            progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_accuracy:.3f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_accuracy:.3f}",
            )

        # Early Stopping
        if val_loss < best_val_loss:
            epochs_without_improvement = 0
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            if save_dir is not None:
                torch.save(best_state, save_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.to("cpu")

    return TrainingHistory(
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
    )
