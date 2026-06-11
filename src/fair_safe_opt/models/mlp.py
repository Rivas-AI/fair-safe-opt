from __future__ import annotations

import torch
from torch import nn


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        
        if input_dim <= 0:
            raise ValueError("Input dimension must be positive.")
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.net[0].in_features:
            raise ValueError("Input feature dimension mismatch.")
        return self.net(x).squeeze(-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return (logits > 0).float()