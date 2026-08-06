import math

import pandas as pd
import pytest

from fair_safe_opt.reporting.robustness_tables import (
    decimals_to_scores,
    seed_averages_and_stddev,
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


def average_group(full_df: pd.DataFrame, group: str = "group_0") -> pd.DataFrame:
    # The reporting pipeline splits by group first, then averages each group's
    # table over seeds -- the order the plotting scripts use, since
    # seed_averages_and_stddev cannot average the string 'group' column.
    return seed_averages_and_stddev(split_by_group(full_df)[group])


def test_decimals_to_scores_scales_and_rounds() -> None:
    assert decimals_to_scores(0.5) == 50.0
    assert decimals_to_scores(0.12345) == 12.35
    assert decimals_to_scores(-0.056) == -5.6


def test_seed_averages_and_stddev_averages_over_seeds_and_adds_std() -> None:
    # Runs on a single group's table (no 'group' column), as the pipeline does.
    df = split_by_group(make_table(seeds=(0, 1)))["group_0"]
    # Make the metric differ across seeds so the mean and std are observable.
    df.loc[df["seed"] == 1, "balanced_accuracy"] += 0.1

    averaged = seed_averages_and_stddev(df)

    assert "seed" not in averaged.columns
    assert "balanced_accuracy_std" in averaged.columns
    assert len(averaged) == 2  # one row per epsilon, seeds collapsed
    row = averaged[averaged["epsilon"] == 0.0]
    assert row["balanced_accuracy"].item() == pytest.approx(0.75)  # mean of 0.7 and 0.8
    assert row["balanced_accuracy_std"].item() == pytest.approx(0.1 / math.sqrt(2))


def test_to_scores_scales_metrics_but_not_epsilon() -> None:
    df = average_group(make_table())

    scores = to_scores(df)

    assert set(scores["epsilon"]) == {0.0, 0.1}
    assert scores.loc[scores["epsilon"] == 0.0, "balanced_accuracy"].item() == 70.0


def test_to_scores_does_not_mutate_input() -> None:
    df = average_group(make_table())
    original = df.copy()

    to_scores(df)

    pd.testing.assert_frame_equal(df, original)


def test_split_by_group_columns_and_row_counts() -> None:
    df = make_table()

    tables = split_by_group(df)

    assert set(tables) == {"combined", "group_0", "group_1"}
    for name, table in tables.items():
        assert "group" not in table.columns
        assert list(table.index) == list(range(len(table)))
        assert len(table) == len(df) / 3
    assert "demographic_parity_difference" in tables["combined"].columns
    assert "demographic_parity_difference" not in tables["group_0"].columns
    assert "equal_opportunity_difference" not in tables["group_1"].columns


def test_values_as_deltas_uses_each_architectures_own_baseline() -> None:
    # After split + average, deltas run per group table: each epsilon > 0 row must
    # be diffed against the epsilon == 0 row of the SAME architecture, not whichever
    # baseline row was iterated last.
    averaged = average_group(make_table(architectures=("logistic", "small_mlp")))

    deltas = values_as_deltas(averaged)

    nonzero = deltas[deltas["epsilon"] > 0]
    assert (nonzero["balanced_accuracy"].round(10) == -0.05).all()


def test_values_as_deltas_works_for_every_group_table() -> None:
    # The eval scripts apply deltas to each per-group averaged table in turn.
    full_df = make_table(architectures=("logistic", "small_mlp"))

    for group in ("combined", "group_0", "group_1"):
        deltas = values_as_deltas(average_group(full_df, group))
        nonzero = deltas[deltas["epsilon"] > 0]
        assert (nonzero["balanced_accuracy"].round(10) == -0.05).all()


def test_values_as_deltas_keeps_baseline_rows_and_input_unchanged() -> None:
    df = average_group(make_table())
    original = df.copy()

    deltas = values_as_deltas(df)

    pd.testing.assert_frame_equal(df, original)
    baseline = deltas[deltas["epsilon"] == 0.0]
    pd.testing.assert_frame_equal(baseline, original[original["epsilon"] == 0.0])
