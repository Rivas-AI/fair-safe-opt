from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TabularDataset:
    x: torch.Tensor
    y: torch.Tensor
    sensitive: torch.Tensor
    continuous_mask: torch.Tensor
    feature_min: torch.Tensor
    feature_max: torch.Tensor


def make_synthetic_tabular(
    n_samples: int = 128,
    n_continuous: int = 3,
    n_categorical: int = 2,
    seed: int = 0,
) -> TabularDataset:
    """Create a tiny tabular binary-classification dataset for tests."""
    rng = np.random.default_rng(seed)
    x_cont = rng.normal(size=(n_samples, n_continuous)).astype("float32")
    sensitive = (x_cont[:, 0] > 0).astype("float32")
    cat_raw = rng.integers(0, n_categorical, size=n_samples)
    x_cat = np.eye(n_categorical, dtype="float32")[cat_raw]
    logits = 1.0 * x_cont[:, 0] - 0.7 * x_cont[:, 1] + 0.3 * sensitive
    y = (logits > np.median(logits)).astype("float32")
    x = np.concatenate([x_cont, x_cat], axis=1)
    continuous_mask = np.array([True] * n_continuous + [False] * n_categorical)
    feature_min = x.copy().min(axis=0).astype("float32")
    feature_max = x.copy().max(axis=0).astype("float32")
    return TabularDataset(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        sensitive=torch.tensor(sensitive, dtype=torch.float32),
        continuous_mask=torch.tensor(continuous_mask, dtype=torch.bool),
        feature_min=torch.tensor(feature_min, dtype=torch.float32),
        feature_max=torch.tensor(feature_max, dtype=torch.float32),
    )
