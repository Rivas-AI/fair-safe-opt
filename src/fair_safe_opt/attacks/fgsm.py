from __future__ import annotations

import torch
from torch import nn

from fair_safe_opt.attacks.masks import clamp_continuous_features, validate_continuous_mask


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    continuous_mask: torch.Tensor,
    feature_min: torch.Tensor | None = None,
    feature_max: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preliminary FGSM scaffold restricted to standardized continuous features.

    The mask and clipping behavior are unit-tested. Later work should add
    attack-strength and evaluation tests before using this for reported results.
    """
    mask = validate_continuous_mask(continuous_mask.to(device=x.device), x.shape[1])
    was_training = model.training
    model.eval()
    x_adv = x.detach().clone().requires_grad_(True)
    loss = nn.functional.binary_cross_entropy_with_logits(model(x_adv), y.float())
    grad = torch.autograd.grad(loss, x_adv)[0]
    perturbation = epsilon * grad.sign() * mask.to(dtype=x.dtype).unsqueeze(0)
    attacked = x_adv.detach() + perturbation
    attacked = clamp_continuous_features(attacked, x, mask, epsilon, feature_min, feature_max)
    model.train(was_training)
    return attacked.detach()
