from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import copy
from pathlib import Path
from tqdm import tqdm

from fair_safe_opt.metrics.classification import balanced_accuracy_from_logits
from fair_safe_opt.training.fair_adversarial import fairness_surrogate
from fair_safe_opt.attacks.pgd import pgd_attack
from fair_safe_opt.training.training_history import TrainingHistory


def train_pgd_madry(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    a_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    a_val: torch.Tensor, 
    continuous_mask: torch.Tensor,
    *,
    out_dir: Path | None = None,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-2,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    steps: int = 10,
    patience: int = 5,
    fairness_metric: str = 'dp'
) -> TrainingHistory:
    '''
    Train a model using adversarial training with PGD attacks and early stopping.
    '''
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    train_losses: list[float] = []
    train_accuracies: list[float] = []
    train_surrogates: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    val_surrogates: list[float] = []
    epochs_without_improvement = 0
    best_val_loss = float("inf")  # initialize to infinity

    progress = tqdm(range(epochs), desc="ERM training", unit="epoch")

    # Epoch loop
    for _ in progress:
        model.train()
        epoch_losses = []
        
        # Iterate through the training data in batches
        for xb, yb in loader:
            adv_xb = pgd_attack(model, xb, yb, epsilon=epsilon, alpha=alpha, steps=steps, continuous_mask=continuous_mask)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(adv_xb), yb.float())
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        model.eval()
        
        x_train_adv = pgd_attack(model, x_train, y_train, epsilon=epsilon, alpha=alpha, steps=steps, continuous_mask=continuous_mask)
        x_val_adv = pgd_attack(model, x_val, y_val, epsilon=epsilon, alpha=alpha, steps=steps, continuous_mask=continuous_mask)
        with torch.no_grad():
            logits_train = model(x_train_adv)
            logits_val = model(x_val_adv)

        # Append training metrics
        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_accuracy = balanced_accuracy_from_logits(logits_train, y_train)
        train_surrogate = fairness_surrogate(logits_train, y_train, a_train, fairness_metric)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_surrogates.append(train_surrogate.item())

        # Calculate Validation loss & acc
        val_loss = nn.functional.binary_cross_entropy_with_logits(logits_val, y_val.float()).item()
        val_accuracy = balanced_accuracy_from_logits(logits_val, y_val)
        val_surrogate = fairness_surrogate(logits_val, y_val, a_val, fairness_metric)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_surrogates.append(val_surrogate.item())

        # Update Progress Bar
        progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            balanced_train_acc=f"{train_accuracy:.3f}",
            train_surrogate=f'{train_surrogate:.4f}',
            val_loss=f"{val_loss:.4f}",
            balanced_val_acc=f"{val_accuracy:.3f}",
            val_surrogate=f'{val_surrogate:.4f}',
        )

        # Early Stopping
        if val_loss < best_val_loss:
            epochs_without_improvement = 0
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            if out_dir is not None:
                torch.save(best_state, out_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)

    return TrainingHistory(
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        train_surrogates=train_surrogates,
        val_surrogates=val_surrogates
    )
