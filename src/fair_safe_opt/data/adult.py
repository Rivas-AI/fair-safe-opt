"""Planned Adult Income preprocessing implementation for Week 2.

Expected input:
    A later explicit call to `load_adult`, which retrieves UCI Adult via
    `fetch_openml(data_id=1590, as_frame=True)`.

Expected output:
    Train, validation, and test arrays; binary target arrays; sensitive-attribute
    arrays; feature names; train-set feature bounds; and a continuous-feature
    mask for attack code.

Status:
    Planned implementation scaffold. This module must not be treated as a
    validated experiment pipeline until Week 2 data-validation tests and summary
    outputs are added. Automated initialization tests must not download Adult.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class AdultSplit:
    x_train: torch.Tensor
    x_val: torch.Tensor
    x_test: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    y_test: torch.Tensor
    a_train: torch.Tensor
    a_val: torch.Tensor
    a_test: torch.Tensor
    continuous_mask: torch.Tensor
    feature_min: torch.Tensor
    feature_max: torch.Tensor
    feature_names: list[str]


def load_adult(
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 0,
    include_sensitive_feature: bool = False,
) -> AdultSplit:
    """Load the planned Adult split structure.

    This function may access the network through OpenML. It is intentionally not
    used by automated tests. Later experiment tasks should add dataset-specific
    validation before treating this as a finalized experiment pipeline.
    """
    frame = fetch_openml(data_id=1590, as_frame=True).frame
    frame = frame.dropna().reset_index(drop=True)

    target_col = "class"
    sensitive_col = "sex"
    y = frame[target_col].astype(str).str.contains(">50K").astype("int64").to_numpy()
    sensitive_raw = frame[sensitive_col].astype(str)
    a = sensitive_raw.str.lower().str.contains("female").astype("int64").to_numpy()

    x_frame = frame.drop(columns=[target_col])
    if not include_sensitive_feature:
        x_frame = x_frame.drop(columns=[sensitive_col])

    categorical_cols = list(x_frame.select_dtypes(include=["category", "object", "str"]).columns)
    continuous_cols = [col for col in x_frame.columns if col not in categorical_cols]

    x_train_val, x_test, y_train_val, y_test, a_train_val, a_test = train_test_split(
        x_frame,
        y,
        a,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val_size = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val, a_train, a_val = train_test_split(
        x_train_val,
        y_train_val,
        a_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), continuous_cols),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )
    pipeline = Pipeline([("preprocessor", preprocessor)])
    x_train_np = pipeline.fit_transform(x_train).astype("float32")
    x_val_np = pipeline.transform(x_val).astype("float32")
    x_test_np = pipeline.transform(x_test).astype("float32")

    cat_names = list(
        pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .get_feature_names_out(categorical_cols)
    )
    feature_names = continuous_cols + cat_names
    continuous_mask = np.array([True] * len(continuous_cols) + [False] * len(cat_names))
    feature_min = x_train_np.min(axis=0)
    feature_max = x_train_np.max(axis=0)

    return AdultSplit(
        x_train=torch.from_numpy(x_train_np),
        x_val=torch.from_numpy(x_val_np),
        x_test=torch.from_numpy(x_test_np),
        y_train=torch.from_numpy(y_train.astype("float32")),
        y_val=torch.from_numpy(y_val.astype("float32")),
        y_test=torch.from_numpy(y_test.astype("float32")),
        a_train=torch.from_numpy(a_train.astype("float32")),
        a_val=torch.from_numpy(a_val.astype("float32")),
        a_test=torch.from_numpy(a_test.astype("float32")),
        continuous_mask=torch.from_numpy(continuous_mask),
        feature_min=torch.from_numpy(feature_min.astype("float32")),
        feature_max=torch.from_numpy(feature_max.astype("float32")),
        feature_names=feature_names,
    )
