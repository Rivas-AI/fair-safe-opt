import pandas as pd


def decimals_to_scores(v: float) -> float:
    return float(f"{v * 100:.2f}")


def seed_averages_and_stddev(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["architecture", "epsilon"]
    metrics = [c for c in df.columns if c not in keys + ["seed"]]

    grouped = df.groupby(keys)[metrics].agg(["mean", "std"])

    # interleave: metric, metric_std, next_metric, next_metric_std, ...
    ordered = []
    for m in metrics:
        ordered.append((m, "mean"))
        ordered.append((m, "std"))
    grouped = grouped[ordered]

    # flatten MultiIndex columns: ('eopp','mean')->'eopp', ('eopp','std')->'eopp_std'
    grouped.columns = [
        metric if stat == "mean" else f"{metric}_std" for metric, stat in grouped.columns
    ]

    return grouped.reset_index()


def to_scores(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["float64"]).columns
    metric_cols = [col for col in numeric_cols if col not in ["epsilon", "seed"]]
    scores_df = df.copy()
    scores_df[metric_cols] = df[metric_cols].map(decimals_to_scores)
    return scores_df


def split_by_group(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    combined_df = df[df["group"] == "combined"].copy()
    combined_df = combined_df.drop("group", axis=1, inplace=False).reset_index(drop=True)

    group0_df = df[df["group"] == "group_0"].copy()
    group0_df = group0_df.drop(
        ["group", "demographic_parity_difference", "equal_opportunity_difference"],
        axis=1,
        inplace=False,
    ).reset_index(drop=True)

    group1_df = df[df["group"] == "group_1"].copy()
    group1_df = group1_df.drop(
        ["group", "demographic_parity_difference", "equal_opportunity_difference"],
        axis=1,
        inplace=False,
    ).reset_index(drop=True)

    return {"combined": combined_df, "group_0": group0_df, "group_1": group1_df}


def values_as_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    baseline_rows = {}

    for idx, row in df.iterrows():
        key = (row["architecture"], row.get("group"))
        if row["epsilon"] == 0.0:
            baseline_rows[key] = row
        else:
            baseline_row = baseline_rows[key]
            for col in df.columns:
                if col not in ["seed", "architecture", "epsilon", "group"]:
                    df.at[idx, col] = row[col] - baseline_row[col]

    return df
