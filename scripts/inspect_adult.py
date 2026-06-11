from fair_safe_opt.data.adult import AdultSplit, load_adult
from fair_safe_opt.metrics.fairness import (
    demographic_parity_difference,
    equal_opportunity_difference,
    demographic_parity_surrogate,
    equal_opportunity_surrogate,
)
import numpy as np
import torch


def print_adult_properties(split: AdultSplit):
    train = split.x_train
    male_mask = split.a_train == 0
    female_mask = split.a_train == 1

    test = split.x_test
    test_0 = test[split.a_test == 0]
    test_1 = test[split.a_test == 1]

    val = split.x_val
    val_0 = val[split.a_val == 0]
    val_1 = val[split.a_val == 1]

    print(f"Num Features: {len(split.feature_names)}\n")

    print(f"Num Training Samples: {train.shape[0]}\n")
    print("Training Set Stats:")
    print(
        f" - # Sensitive Group 0 (male): {train[male_mask].shape[0]}  (~{int(train[male_mask].shape[0] / train.shape[0] * 100)}%)"
    )
    print(f"     - Positive Prediction Rate: {(split.y_train[male_mask] == 1).mean()}")
    print(
        f" - # Sensitive Group 1 (female): {train[female_mask].shape[0]}  (~{int(train[female_mask].shape[0] / train.shape[0] * 100)}%)"
    )
    print(f"     - Positive Prediction Rate: {(split.y_train[female_mask] == 1).mean()}")

    print()
    print(f"Num Test Samples: {test.shape[0]}")
    print("Test Set Stats:")
    print(
        f" - # Sensitive Group 0 (male): {test_0.shape[0]}  (~{int(test_0.shape[0] / test.shape[0] * 100)}%)"
    )
    print(f"     - Positive Prediction Rate: {(split.y_test[split.a_test == 0] == 1).mean()}")
    print(
        f" - # Sensitive Group 1 (female): {test_1.shape[0]}  (~{int(test_1.shape[0] / test.shape[0] * 100)}%)"
    )
    print(f"     - Positive Prediction Rate: {(split.y_test[split.a_test == 1] == 1).mean()}")

    print()
    print(f"Num Val Samples: {val.shape[0]}")
    print("Val Set Stats:")
    print(
        f" - # Sensitive Group 0 (male): {val_0.shape[0]}  (~{int(val_0.shape[0] / val.shape[0] * 100)}%)"
    )
    print(f"     - Positive Prediction Rate: {(split.y_val[split.a_val == 0] == 1).mean()}")
    print(
        f" - # Sensitive Group 1 (female): {val_1.shape[0]}  (~{int(val_1.shape[0] / val.shape[0] * 100)}%)"
    )
    print(f"     - Positive Prediction Rate: {(split.y_val[split.a_val == 1] == 1).mean()}")
    print()


def dummy_ones_classifier(features) -> np.ndarray:
    # This is a dummy classifier that always predicts 1
    return np.ones(features.shape[0], dtype=int)


def dummy_unfair_classifier(features, sensitive_attr) -> np.ndarray:
    preds = np.zeros(features.shape[0], dtype=int)

    mask = sensitive_attr == 0  # Predict 1 for group 0, 0 for group 1
    preds[mask] = 1

    return preds


def test_fairness_with_dummies(split):
    # Generate fake predictions
    ones_preds = torch.tensor(dummy_ones_classifier(split.x_test), dtype=torch.float32)
    unfair_preds = torch.tensor(
        dummy_unfair_classifier(split.x_test, split.a_test), dtype=torch.float32
    )

    # Convert sensitive/label array to tensor
    a_test_tensor = torch.tensor(split.a_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(split.y_test, dtype=torch.float32)

    # Evaluate fairness metrics for ones classifier
    dpd_ones = demographic_parity_difference(ones_preds, a_test_tensor)
    eod_ones = equal_opportunity_difference(y_test_tensor, ones_preds, a_test_tensor)
    dpd_ones_surrogate = demographic_parity_surrogate(ones_preds, a_test_tensor)
    eod_ones_surrogate = equal_opportunity_surrogate(ones_preds, y_test_tensor, a_test_tensor)

    print("Fairness Metrics for All Ones Dummy on Test Dataset: ")
    print(f"    - Demographic Parity Difference: {dpd_ones}")
    print(f"    - Equal Opportunity Difference: {eod_ones}")
    print(f"    - Demographic Parity Surrogate: {dpd_ones_surrogate}")
    print(f"    - Equal Opportunity Surrogate: {eod_ones_surrogate}")

    # Evaluate fairness metrics for unfair classifier
    dpd_unfair = demographic_parity_difference(unfair_preds, a_test_tensor)
    eod_unfair = equal_opportunity_difference(y_test_tensor, unfair_preds, a_test_tensor)
    dpd_unfair_surrogate = demographic_parity_surrogate(unfair_preds, a_test_tensor)
    eod_unfair_surrogate = equal_opportunity_surrogate(unfair_preds, y_test_tensor, a_test_tensor)

    print("Fairness Metrics for Unfair Dummy on Test Dataset: ")
    print(f"    - Demographic Parity Difference: {dpd_unfair}")
    print(f"    - Equal Opportunity Difference: {eod_unfair}")
    print(f"    - Demographic Parity Surrogate: {dpd_unfair_surrogate}")
    print(f"    - Equal Opportunity Surrogate: {eod_unfair_surrogate}")


adult_split = load_adult()
print_adult_properties(adult_split)
test_fairness_with_dummies(adult_split)
