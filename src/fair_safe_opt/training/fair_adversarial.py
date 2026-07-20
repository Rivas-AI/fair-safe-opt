"""Planned fair-adversarial training helpers for Weeks 6 and 7.

Expected input:
    Binary logits, labels, sensitive-group indicators, selected fairness
    surrogate (`dp` or `eopp`), and a penalty weight.

Expected output:
    Differentiable surrogate or penalized loss tensor for later training-loop
    integration.

Status:
    Planned implementation scaffold. These helpers do not implement the full
    fairness-constrained or fairness-penalized adversarial-training method. Week
    6 and Week 7 must add training-loop tests, configs, logging, and method
    documentation before use in reported experiments.
"""

from __future__ import annotations

import torch
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import copy

from fair_safe_opt.metrics.fairness import (
    demographic_parity_surrogate,
    equal_opportunity_surrogate,
)

from fair_safe_opt.training.training_history import TrainingHistory
from fair_safe_opt.attacks.pgd import pgd_attack
from fair_safe_opt.evaluation.evaluate import balanced_accuracy_from_logits

def fairness_surrogate(
    logits: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    if metric == "dp":
        return demographic_parity_surrogate(probabilities, sensitive)
    if metric == "eopp":
        return equal_opportunity_surrogate(probabilities, y, sensitive)
    raise ValueError(f"Unsupported fairness metric: {metric}")

def fairness_penalized_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    *,
    metric: str = "dp",
    tolerance: float = 0.0,
    multiplier: float = 1.0,
    penalty_weight: float = 1.0,
) -> torch.Tensor:
    """Preliminary differentiable fairness-penalty helper.

    This helper is not a complete fairness-constrained adversarial-training
    implementation. Later work must add training-loop tests and documentation
    before using it for reported experiments.
    """
    base_loss = nn.functional.binary_cross_entropy_with_logits(logits, y.float())
    surrogate = fairness_surrogate(logits, y, sensitive, metric)
    constraint = surrogate - tolerance
    
    multiplier_term = multiplier * constraint
    
    penalty_term = (penalty_weight / 2) * torch.clamp(constraint, min=0) ** 2
    
    return base_loss + multiplier_term + penalty_term

def fairness_constrained_pgd(
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
    fairness_metric: str = 'dp',
    tolerance: float = 0.0,
    multiplier: float = 0.0,
    penalty_weight: float = 1.0,   # CHANGED: 0.0 made the penalty term a no-op
    dual_lr: float = 1.0,          # NEW: eta for the once-per-epoch lambda update
) -> TrainingHistory:
    '''
    Adversarially train a model using fairness constraints in the loss function
    to make a fair and adversarially robust model.
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(x_train, y_train, a_train), batch_size=batch_size, shuffle=True)
    train_losses: list[float] = []
    train_accuracies: list[float] = []
    train_fairnesses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    val_fairnesses: list[float] = []
    epochs_without_improvement = 0
    best_val_loss = float("inf")
    best_val_fairness = float('inf')
    prev_violation = float('inf')  # NEW: tracks constraint violation for the rho schedule

    progress = tqdm(range(epochs), desc="Fair-adv training", unit="epoch")  # CHANGED: desc said "ERM training"

    # Epoch loop
    for _ in progress:
        model.train()
        epoch_losses = []
        epoch_fairnesses = []

        # Iterate through the training data in batches
        for xb, yb, ab in loader:
            adv_xb = pgd_attack(model, xb, yb, epsilon=epsilon, alpha=alpha, steps=steps, continuous_mask=continuous_mask)
            optimizer.zero_grad(set_to_none=True)
            logits = model(adv_xb)  # NEW: forward once, reuse below (was two forward passes)
            surrogate = fairness_surrogate(logits, yb, ab, metric=fairness_metric)
            loss = fairness_penalized_loss(logits, yb, ab, metric=fairness_metric, tolerance=tolerance, multiplier=multiplier, penalty_weight=penalty_weight)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            epoch_fairnesses.append(float(surrogate.item()))
        model.eval()

        x_train_adv = pgd_attack(model, x_train, y_train, epsilon=epsilon, alpha=alpha, steps=steps, continuous_mask=continuous_mask)
        x_val_adv = pgd_attack(model, x_val, y_val, epsilon=epsilon, alpha=alpha, steps=steps, continuous_mask=continuous_mask)
        with torch.no_grad():
            logits_train = model(x_train_adv)
            logits_val = model(x_val_adv)

        # NEW: augmented Lagrangian updates, once per epoch, on the full-train estimate
        with torch.no_grad():
            full_surrogate = fairness_surrogate(logits_train, y_train, a_train, metric=fairness_metric)
        violation = full_surrogate.item() - tolerance                 # NEW: G(theta)
        multiplier = max(0.0, multiplier + dual_lr * violation)       # NEW: dual ascent, clamped at 0
        if violation > 0 and violation > 0.25 * prev_violation:       # NEW: raise rho only if violation
            penalty_weight *= 2                                       # NEW: isn't shrinking fast enough
        prev_violation = max(violation, 0.0)                          # NEW
        # CHANGED: replaces unconditional `penalty_weight += 1`

        # Append training metrics
        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_fairness = sum(epoch_fairnesses) / len(epoch_fairnesses)
        train_accuracy = balanced_accuracy_from_logits(logits_train, y_train)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_fairnesses.append(train_fairness)

        # Calculate Validation loss & acc
        val_loss = nn.functional.binary_cross_entropy_with_logits(logits_val, y_val.float()).item()
        val_accuracy = balanced_accuracy_from_logits(logits_val, y_val)
        val_fairness = fairness_surrogate(logits_val, y_val, a_val, metric=fairness_metric).item()  # CHANGED: .item() — was storing/comparing a tensor
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_fairnesses.append(val_fairness)

        # Update Progress Bar
        progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            balanced_train_acc=f"{train_accuracy:.3f}",
            train_fairness=f'{train_fairness:.3f}',
            val_loss=f"{val_loss:.4f}",
            balanced_val_acc=f"{val_accuracy:.3f}",
            val_fairness=f'{val_fairness:.3f}',
            lam=f'{multiplier:.3f}',   # NEW: watching lambda evolve is your main diagnostic
        )

        # Early Stopping
        if val_loss < best_val_loss and val_fairness < best_val_fairness:
            epochs_without_improvement = 0
            best_val_loss = val_loss
            best_val_fairness = val_fairness
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
        train_fairnesses=train_fairnesses,  # CHANGED: was passing the scalar train_fairness
        val_fairnesses=val_fairnesses,      # CHANGED: was passing the scalar val_fairness
    )