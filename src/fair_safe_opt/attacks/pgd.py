"""Planned PGD attack implementation for Week 4.

Expected input:
    A binary PyTorch classifier, feature tensor, binary labels, epsilon, step
    size, number of steps, a boolean continuous-feature mask, and optional
    train-set feature bounds.

Expected output:
    A projected perturbed feature tensor with categorical and one-hot positions
    unchanged.

Status:
    Planned implementation scaffold. Mask and clipping behavior are covered by
    offline tests, but attack-strength and evaluation tests are deferred to Week
    4 before this can support reported robust-accuracy results.
"""

from __future__ import annotations

import torch
from torch import nn

from fair_safe_opt.attacks.masks import clamp_continuous_features, validate_continuous_mask


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    continuous_mask: torch.Tensor,
    feature_min: torch.Tensor | None = None,
    feature_max: torch.Tensor | None = None,
    random_start: bool = False,
) -> torch.Tensor:
    """Preliminary PGD scaffold restricted to standardized continuous features.

    The mask and clipping behavior are unit-tested. Later work should add
    attack-strength and evaluation tests before using this for reported results.
    """
    mask = validate_continuous_mask(continuous_mask.to(device=x.device), x.shape[1])
    mask_float = mask.to(dtype=x.dtype).unsqueeze(0)
    was_training = model.training
    model.eval()
    if random_start:
        noise = torch.empty_like(x).uniform_(-epsilon, epsilon) * mask_float
        x_adv = clamp_continuous_features(x + noise, x, mask, epsilon, feature_min, feature_max)
    else:
        x_adv = x.detach().clone()

    for _ in range(steps):
        x_adv = x_adv.detach().clone().requires_grad_(True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x_adv), y.float())
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign() * mask_float
        x_adv = clamp_continuous_features(x_adv, x, mask, epsilon, feature_min, feature_max)

    model.train(was_training)
    return x_adv.detach()
