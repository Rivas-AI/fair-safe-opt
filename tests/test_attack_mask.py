import torch

from fair_safe_opt.attacks.fgsm import fgsm_attack
from fair_safe_opt.attacks.masks import validate_continuous_mask
from fair_safe_opt.attacks.pgd import pgd_attack
from fair_safe_opt.data.synthetic import make_synthetic_tabular
from fair_safe_opt.models import LogisticRegression


def test_validate_continuous_mask_rejects_wrong_shape() -> None:
    mask = torch.tensor([[True, False]])

    try:
        validate_continuous_mask(mask, 2)
    except ValueError:
        return

    raise AssertionError("Expected mask validation to fail")


def test_fgsm_changes_continuous_features_only() -> None:
    dataset = make_synthetic_tabular(n_samples=32, seed=4)
    model = LogisticRegression(dataset.x.shape[1])

    attacked = fgsm_attack(
        model,
        dataset.x,
        dataset.y,
        epsilon=0.05,
        continuous_mask=dataset.continuous_mask,
        feature_min=dataset.feature_min,
        feature_max=dataset.feature_max,
    )

    categorical = ~dataset.continuous_mask
    assert torch.allclose(attacked[:, categorical], dataset.x[:, categorical])
    assert (
        torch.max(
            torch.abs(attacked[:, dataset.continuous_mask] - dataset.x[:, dataset.continuous_mask])
        )
        <= 0.05001
    )
    assert torch.all(attacked[:, dataset.continuous_mask] <= dataset.feature_max[dataset.continuous_mask])
    assert torch.all(attacked[:, dataset.continuous_mask] >= dataset.feature_min[dataset.continuous_mask])


def test_pgd_changes_continuous_features_only() -> None:
    dataset = make_synthetic_tabular(n_samples=32, seed=5)
    model = LogisticRegression(dataset.x.shape[1])

    attacked = pgd_attack(
        model,
        dataset.x,
        dataset.y,
        epsilon=0.05,
        alpha=0.02,
        steps=3,
        continuous_mask=dataset.continuous_mask,
        feature_min=dataset.feature_min,
        feature_max=dataset.feature_max,
    )

    categorical = ~dataset.continuous_mask
    assert torch.allclose(attacked[:, categorical], dataset.x[:, categorical])
    assert (
        torch.max(
            torch.abs(attacked[:, dataset.continuous_mask] - dataset.x[:, dataset.continuous_mask])
        )
        <= 0.05001
    )
    assert torch.all(attacked[:, dataset.continuous_mask] <= dataset.feature_max[dataset.continuous_mask])
    assert torch.all(attacked[:, dataset.continuous_mask] >= dataset.feature_min[dataset.continuous_mask])
