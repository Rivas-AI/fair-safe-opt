import math
import warnings

import torch

from fair_safe_opt.metrics.fairness import (
    demographic_parity_difference,
    demographic_parity_surrogate,
    equal_opportunity_difference,
    equal_opportunity_surrogate,
)


def test_demographic_parity_difference_balanced() -> None:
    y_pred = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    sensitive = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    assert demographic_parity_difference(y_pred, sensitive) == 0.0


def test_demographic_parity_difference_nonzero() -> None:
    y_pred = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    sensitive = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    assert demographic_parity_difference(y_pred, sensitive) == 1.0


def test_equal_opportunity_difference() -> None:
    y_true = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    y_pred = torch.tensor([1, 0, 1, 1], dtype=torch.float32)
    sensitive = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    assert equal_opportunity_difference(y_true, y_pred, sensitive) == 0.5


def test_surrogates_are_squared_probability_gaps() -> None:
    probs = torch.tensor([0.2, 0.4, 0.8, 1.0])
    y_true = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    sensitive = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    assert torch.isclose(demographic_parity_surrogate(probs, sensitive), torch.tensor(0.36))
    assert torch.isclose(equal_opportunity_surrogate(probs, y_true, sensitive), torch.tensor(0.36))


def test_missing_positive_group_warns_and_returns_nan() -> None:
    y_true = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
    y_pred = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    sensitive = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = equal_opportunity_difference(y_true, y_pred, sensitive)

    assert math.isnan(value)
    assert caught
