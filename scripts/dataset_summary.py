"""
dataset_summary.py
----------------------------------
Compute the numbers for the Adult dataset-summary table in the paper.

The processed arrays come from the same load path training uses
(fair_safe_opt.data.adult.load_adult), so the row filtering, missing-value
handling, sensitive-attribute encoding, and stratified split all match what the
models actually see. A couple of the requested metadata figures (how many rows
were dropped for missing values and from which columns, and the feature count
*before* one-hot encoding) are not carried on the returned AdultSplit, so we also
read the raw OpenML frame once. fetch_openml caches locally, so this is a cheap
re-read of the same source load_adult pulls from, not a re-implementation of the
preprocessing.

Output is plain text. Percentages are printed to 1 decimal, rates to 3 decimals.
"""

from __future__ import annotations

import numpy as np
import torch
import yaml
from sklearn.datasets import fetch_openml

from fair_safe_opt.data.adult import load_adult, AdultSplit
from fair_safe_opt.utils.paths import TRAINING_CONFIG_PATH

# The dataset load_adult pulls from (UCI Adult on OpenML). Kept in sync by hand
# with fair_safe_opt.data.adult; used only to describe the raw frame, never to
# reproduce the processed features.
_ADULT_OPENML_ID = 1590
_TARGET_COL = "class"
_SENSITIVE_COL = "sex"


def _combine(*tensors: torch.Tensor) -> np.ndarray:
    """Concatenate split tensors back into the full-dataset array as int."""
    return torch.cat(tensors).round().to(torch.int64).numpy()


def group_summary(a: np.ndarray, y: np.ndarray) -> None:
    """Per-group n, share of total, Pr{Y=1}, and count of Y=1 for the full set."""
    total = a.shape[0]
    groups = [
        ("G0 (a==0, male)  ", a == 0),
        ("G1 (a==1, female)", a == 1),
        ("All              ", np.ones_like(a, dtype=bool)),
    ]
    print("Full dataset (train + val + test combined)")
    print(f"{'group':<18}{'n':>8}{'% total':>10}{'Pr{Y=1}':>10}{'#Y=1':>8}")
    for label, mask in groups:
        n = int(mask.sum())
        y_g = y[mask]
        pct = 100.0 * n / total
        rate = float(y_g.mean()) if n else 0.0
        pos = int(y_g.sum())
        print(f"{label:<18}{n:>8}{pct:>9.1f}%{rate:>10.3f}{pos:>8}")


def cell_counts(a: np.ndarray, y: np.ndarray) -> None:
    """The four (A, Y) joint cell counts."""
    print("(A, Y) cell counts")
    for a_val in (0, 1):
        for y_val in (0, 1):
            n = int(((a == a_val) & (y == y_val)).sum())
            print(f"  (A={a_val}, Y={y_val}): {n}")


def split_sizes(split: AdultSplit) -> None:
    """Train/val/test sizes and the stratification variable."""
    n_train = split.y_train.shape[0]
    n_val = split.y_val.shape[0]
    n_test = split.y_test.shape[0]
    total = n_train + n_val + n_test
    print("Split sizes")
    print(f"  train: {n_train} ({100.0 * n_train / total:.1f}%)")
    print(f"  val:   {n_val} ({100.0 * n_val / total:.1f}%)")
    print(f"  test:  {n_test} ({100.0 * n_test / total:.1f}%)")
    print(
        "  stratified on: Y (the class label), at both the "
        "train+val/test and train/val split stages"
    )


def feature_report(split: AdultSplit, n_raw_features: int) -> None:
    """Feature counts before/after one-hot, and whether fnlwgt is a feature."""
    n_after = len(split.feature_names)
    has_fnlwgt = "fnlwgt" in split.feature_names
    print("Features")
    print(f"  before one-hot encoding: {n_raw_features}")
    print(f"  after one-hot encoding:  {n_after}")
    print(f"  fnlwgt included as a feature: {'yes' if has_fnlwgt else 'no'}")


def missing_and_raw_features(
    data_id: int, target_col: str, sensitive_col: str, include_sensitive_feature: bool
) -> int:
    """Report rows dropped for missing values and return the pre-one-hot feature count.

    Mirrors load_adult's scope: it calls frame.dropna() over the whole frame, and
    (with include_sensitive_feature False) feeds every column except the target and
    the sensitive attribute into the encoder. Returns that raw feature-column count.
    """
    frame = fetch_openml(data_id=data_id, as_frame=True).frame
    n_before = len(frame)
    per_col = frame.isna().sum()
    dropped = int((frame.isna().any(axis=1)).sum())

    print("Missing values (dropna over the full frame, as in load_adult)")
    print(f"  rows before dropna: {n_before}")
    print(f"  rows after dropna:  {n_before - dropped}")
    print(f"  rows dropped:       {dropped}")
    offending = per_col[per_col > 0]
    if offending.empty:
        print("  columns with missing values: none")
    else:
        print("  columns with missing values (NaN count in raw frame):")
        for col, cnt in offending.items():
            print(f"    {col}: {int(cnt)}")

    drop_cols = [target_col]
    if not include_sensitive_feature:
        drop_cols.append(sensitive_col)
    return len([c for c in frame.columns if c not in drop_cols])


def summarize(seed: int, data_id: int, target_col: str, sensitive_col: str) -> None:
    """Run every section using plain values only."""
    # Full-dataset group/cell/rate numbers are invariant to the split seed (the
    # split only partitions the same rows); the seed just fixes a concrete split.
    split = load_adult(random_state=seed)
    a = _combine(split.a_train, split.a_val, split.a_test)
    y = _combine(split.y_train, split.y_val, split.y_test)

    group_summary(a, y)
    print()
    cell_counts(a, y)
    print()
    split_sizes(split)
    print()
    n_raw_features = missing_and_raw_features(
        data_id, target_col, sensitive_col, include_sensitive_feature=False
    )
    print()
    feature_report(split, n_raw_features)


def main() -> None:
    with open(TRAINING_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    seed = int(config["seed_sweep"][0])
    summarize(
        seed=seed,
        data_id=_ADULT_OPENML_ID,
        target_col=_TARGET_COL,
        sensitive_col=_SENSITIVE_COL,
    )


if __name__ == "__main__":
    main()
