import torch

from fair_safe_opt.evaluation.evaluate import create_metric_table


def test_create_metric_table_index_and_columns() -> None:
    preds = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    probs = torch.tensor([0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    sensitive = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    table = create_metric_table(preds, probs, labels, sensitive)

    assert list(table.index) == ["group_0", "group_1", "combined"]
    assert list(table.columns) == [
        "balanced_accuracy",
        "auc",
        "advantaged tpr",
        "demographic_parity_difference",
        "equal_opportunity_difference",
    ]


def test_create_metric_table_group_values_consistent_with_combined() -> None:
    # Group 0 predicts perfectly, group 1 predicts every label wrong.
    preds = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    probs = torch.tensor([0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    sensitive = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    table = create_metric_table(preds, probs, labels, sensitive)

    assert table.loc["group_0", "balanced_accuracy"] == 1.0
    assert table.loc["group_1", "balanced_accuracy"] == 0.0
    assert table.loc["combined", "balanced_accuracy"] == 0.5

    assert table.loc["group_0", "auc"] == 1.0
    assert table.loc["group_1", "auc"] == 0.0
    assert table.loc["combined", "auc"] == 0.5
