import pandas as pd


def decimals_to_scores(v: float) -> float:
    return float(f"{v * 100:.2f}")


def average_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['architecture', 'epsilon', 'group']).mean().reset_index().drop('seed', axis=1)


def to_scores(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=['float64']).columns
    metric_cols = [col for col in numeric_cols if col not in ['epsilon', 'seed']]
    scores_df = df.copy()
    scores_df[metric_cols] = df[metric_cols].map(decimals_to_scores)
    return scores_df


def split_by_group(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    combined_df = df[df['group'] == 'combined'].copy()
    combined_df = combined_df.drop('group', axis=1, inplace=False).reset_index(drop=True)

    group0_df = df[df['group'] == 'group_0'].copy()
    group0_df = group0_df.drop(['group', 'demographic_parity_difference', 'equal_opportunity_difference'], axis=1, inplace=False).reset_index(drop=True)

    group1_df = df[df['group'] == 'group_1'].copy()
    group1_df = group1_df.drop(['group', 'demographic_parity_difference', 'equal_opportunity_difference'], axis=1, inplace=False).reset_index(drop=True)

    return {'combined': combined_df, 'group_0': group0_df, 'group_1': group1_df}


def values_as_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    baseline_row = pd.Series()

    for idx, row in df.iterrows():
        if row['epsilon'] == 0.0:
            baseline_row = row
        else:
            for col in df.columns:
                if col not in ['seed', 'architecture', 'epsilon', 'group']:
                    df.at[idx, col] = row[col] - baseline_row[col]

    return df
