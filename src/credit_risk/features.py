"""Column groups and the preprocessing that every model shares."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk import config
from credit_risk.data import encode_target, prepare_frame


def feature_lists(drop: Sequence[str] = ()) -> tuple[list[str], list[str]]:
    """Numeric and categorical columns, optionally dropping protected fields."""
    dropped = set(drop)
    numeric = [column for column in config.NUMERIC_FEATURES if column not in dropped]
    categorical = [column for column in config.CATEGORICAL_FEATURES if column not in dropped]
    return numeric, categorical


def ordered_levels(column: str, observed: Sequence[str]) -> list[str]:
    """Codebook order for forms. Unknown labels are appended alphabetically."""
    observed_set = {str(level) for level in observed if pd.notna(level)}
    preferred = [level for level in config.CATEGORY_ORDERS.get(column, []) if level in observed_set]
    extras = sorted(observed_set - set(preferred))
    return preferred + extras


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean ``frame`` and return features plus a 0/1 default indicator."""
    prepared = prepare_frame(frame)
    features = prepared[config.FEATURE_COLUMNS].copy()
    target = encode_target(prepared[config.TARGET_COLUMN])
    features.index = pd.RangeIndex(len(features))
    target.index = features.index
    return features, target


def build_preprocessor(
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Impute, scale numerics, and one-hot encode categories.

    The same transformer is inside every model pipeline, so logistic regression
    and the tree models see the same representation. Unknown categories at
    score time become a row of zeros rather than an error.
    """
    if numeric_features is None or categorical_features is None:
        default_numeric, default_categorical = feature_lists()
        numeric = list(default_numeric if numeric_features is None else numeric_features)
        categorical = list(
            default_categorical if categorical_features is None else categorical_features
        )
    else:
        numeric = list(numeric_features)
        categorical = list(categorical_features)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def category_codebook(features: pd.DataFrame) -> dict[str, list[str]]:
    """Observed category levels in display order."""
    return {
        column: ordered_levels(column, features[column].dropna().unique())
        for column in config.CATEGORICAL_FEATURES
        if column in features.columns
    }


def numeric_bounds(features: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Min, max, and median for sliders and validation context."""
    bounds: dict[str, dict[str, float]] = {}
    for column in config.NUMERIC_FEATURES:
        series = features[column]
        bounds[column] = {
            "min": float(np.nanmin(series.to_numpy())),
            "max": float(np.nanmax(series.to_numpy())),
            "median": float(np.nanmedian(series.to_numpy())),
        }
    return bounds


def reference_applicant(features: pd.DataFrame) -> dict[str, float | str]:
    """A typical training applicant: median numerics and the most common label."""
    profile: dict[str, float | str] = {}
    for column in config.NUMERIC_FEATURES:
        series = features[column].dropna()
        median = float(series.median())
        if np.all(np.isclose(series.to_numpy() % 1, 0)):
            median = float(np.round(median))
        median = float(np.clip(median, series.min(), series.max()))
        profile[column] = median
    for column in config.CATEGORICAL_FEATURES:
        mode = features[column].mode(dropna=True)
        profile[column] = str(mode.iloc[0])
    return profile
