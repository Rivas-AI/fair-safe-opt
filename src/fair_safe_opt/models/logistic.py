from __future__ import annotations

import torch
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("Input dimension must be positive.")
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.linear.in_features:
            raise ValueError("Input feature dimension mismatch.")
        return self.linear(x).squeeze(-1)

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return (logits > 0).float()
