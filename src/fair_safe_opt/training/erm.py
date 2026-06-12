from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fair_safe_opt.metrics.classification import accuracy_from_logits


@dataclass(frozen=True)
class TrainingHistory:
    losses: list[float]
    accuracies: list[float]


def set_device():
    '''
    Sets device used for training. Heirarchy goes CUDA, MPS, then CPU if none are available. 
    returns:
    - (torch.device): device to be used.
    '''
    if torch.cuda.is_available():
        print("Using CUDA")
        return(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        print("Using MPS")
        return(torch.device("mps"))
    else:
        print("Using CPU")
        return(torch.device("cpu"))
    
def train_erm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-2,
) -> TrainingHistory:
    device = set_device()
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    losses: list[float] = []
    accuracies: list[float] = []

    for i in range(epochs):
        print(f"Epoch {i + 1}/{epochs}")
        model = model.to(device)
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
            logits = model(x)
        losses.append(sum(epoch_losses) / len(epoch_losses))
        accuracies.append(accuracy_from_logits(logits, y))

    model.to("cpu")
    return TrainingHistory(losses=losses, accuracies=accuracies)
