from __future__ import annotations

import torch


def validate_continuous_mask(mask: torch.Tensor, n_features: int) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise TypeError("continuous feature mask must be boolean")
    if mask.ndim != 1:
        raise ValueError("continuous feature mask must be one-dimensional")
    if mask.numel() != n_features:
        raise ValueError("continuous feature mask length must match feature dimension")
    if not bool(mask.any().item()):
        raise ValueError("continuous feature mask must select at least one feature")
    return mask


def clamp_continuous_features(
    x: torch.Tensor,
    original_x: torch.Tensor,
    continuous_mask: torch.Tensor,
    epsilon: float,
    feature_min: torch.Tensor | None = None,
    feature_max: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = validate_continuous_mask(continuous_mask.to(device=x.device), x.shape[1])
    lower = original_x - epsilon
    upper = original_x + epsilon
    if feature_min is not None:
        lower = torch.maximum(lower, feature_min.to(device=x.device, dtype=x.dtype))
    if feature_max is not None:
        upper = torch.minimum(upper, feature_max.to(device=x.device, dtype=x.dtype))
    clipped_cont = torch.minimum(torch.maximum(x, lower), upper)
    return torch.where(mask.unsqueeze(0), clipped_cont, original_x)
