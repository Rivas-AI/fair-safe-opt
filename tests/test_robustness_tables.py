import pandas as pd
import pytest

from fair_safe_opt.reporting.robustness_tables import (
    decimals_to_scores,
    average_over_seeds,
    to_scores,
    split_by_group,
    values_as_deltas,
)


def make_table(architectures=("logistic",), epsilons=(0.0, 0.1), seeds=(0, 1)) -> pd.DataFrame:
    # Synthetic full_df shaped like the eval scripts' output: one row per
    # (architecture, epsilon, seed, group), accuracy drops by 0.05 per 0.1 epsilon.
    rows = []
    for arch_offset, architecture in enumerate(architectures):
        for epsilon in epsilons:
            for seed in seeds:
                for group_offset, group in enumerate(["group_0", "group_1", "combined"]):
                    rows.append(
                        {
                            "architecture": architecture,
                            "epsilon": epsilon,
                            "seed": seed,
                            "group": group,
                            "balanced_accuracy": 0.7
                            + 0.1 * group_offset
                            + 0.02 * arch_offset
                            - 0.5 * epsilon,
                            "demographic_parity_difference": 0.1,
                            "equal_opportunity_difference": 0.2,
                        }
                    )
    return pd.DataFrame(rows)


def test_decimals_to_scores_scales_and_rounds() -> None:
    assert decimals_to_scores(0.5) == 50.0
    assert decimals_to_scores(0.12345) == 12.35
    assert decimals_to_scores(-0.056) == -5.6


def test_average_over_seeds_averages_and_drops_seed() -> None:
    df = make_table(seeds=(0, 1))
    # Make the metric differ across seeds so the mean is observable.
    df.loc[df["seed"] == 1, "balanced_accuracy"] += 0.1

    averaged = average_over_seeds(df)

    assert "seed" not in averaged.columns
    assert len(averaged) == len(df) / 2
    row = averaged[(averaged["epsilon"] == 0.0) & (averaged["group"] == "group_0")]
    assert row["balanced_accuracy"].item() == pytest.approx(0.75)


def test_to_scores_scales_metrics_but_not_epsilon() -> None:
    df = average_over_seeds(make_table())

    scores = to_scores(df)

    assert set(scores["epsilon"]) == {0.0, 0.1}
    assert (
        scores.loc[
            (scores["epsilon"] == 0.0) & (scores["group"] == "group_0"), "balanced_accuracy"
        ].item()
        == 70.0
    )


def test_to_scores_does_not_mutate_input() -> None:
    df = average_over_seeds(make_table())
    original = df.copy()

    to_scores(df)

    pd.testing.assert_frame_equal(df, original)


def test_split_by_group_columns_and_row_counts() -> None:
    df = average_over_seeds(make_table())

    tables = split_by_group(df)

    assert set(tables) == {"combined", "group_0", "group_1"}
    for name, table in tables.items():
        assert "group" not in table.columns
        assert list(table.index) == list(range(len(table)))
        assert len(table) == len(df) / 3
    assert "demographic_parity_difference" in tables["combined"].columns
    assert "demographic_parity_difference" not in tables["group_0"].columns
    assert "equal_opportunity_difference" not in tables["group_1"].columns


def test_values_as_deltas_uses_each_groups_own_baseline() -> None:
    # Regression test: pre-split tables contain all three groups, and each
    # epsilon > 0 row must be diffed against the epsilon == 0 row of the SAME
    # architecture and group, not whichever baseline row was iterated last.
    df = average_over_seeds(make_table(architectures=("logistic", "small_mlp")))

    deltas = values_as_deltas(df)

    nonzero = deltas[deltas["epsilon"] > 0]
    assert (nonzero["balanced_accuracy"].round(10) == -0.05).all()


def test_values_as_deltas_works_after_split_by_group() -> None:
    # The eval scripts apply deltas to per-group tables with no 'group' column.
    df = average_over_seeds(make_table(architectures=("logistic", "small_mlp")))

    for table in split_by_group(df).values():
        deltas = values_as_deltas(table)
        nonzero = deltas[deltas["epsilon"] > 0]
        assert (nonzero["balanced_accuracy"].round(10) == -0.05).all()


def test_values_as_deltas_keeps_baseline_rows_and_input_unchanged() -> None:
    df = average_over_seeds(make_table())
    original = df.copy()

    deltas = values_as_deltas(df)

    pd.testing.assert_frame_equal(df, original)
    baseline = deltas[deltas["epsilon"] == 0.0]
    pd.testing.assert_frame_equal(baseline, original[original["epsilon"] == 0.0])
