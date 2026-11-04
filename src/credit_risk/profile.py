"""Default rates by segment, and which columns the importance table puts first."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk import config


def categorical_default_rates(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Applicant count and default rate for every observed category level."""
    actual = np.asarray(target).astype(int)
    frame = features.copy()
    frame["_default"] = actual
    rows: list[dict[str, object]] = []
    for column in config.CATEGORICAL_FEATURES:
        if column not in frame.columns:
            continue
        for level, part in frame.groupby(column, dropna=False):
            label = "missing" if pd.isna(level) else str(level)
            rows.append(
                {
                    "feature": column,
                    "level": label,
                    "n": int(len(part)),
                    "defaults": int(part["_default"].sum()),
                    "default_rate": float(part["_default"].mean()) if len(part) else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["feature", "default_rate"], ascending=[True, False]).reset_index(
        drop=True
    )


def numeric_band_default_rates(
    features: pd.DataFrame, target: pd.Series, q: int = 4
) -> pd.DataFrame:
    """Default rate by quantile band for each numeric column."""
    actual = np.asarray(target).astype(int)
    rows: list[dict[str, object]] = []
    for column in config.NUMERIC_FEATURES:
        if column not in features.columns:
            continue
        series = features[column]
        usable = series.notna()
        distinct = int(series.loc[usable].nunique())
        if distinct < 2:
            continue
        bands = pd.qcut(series, q=min(q, distinct), duplicates="drop")
        grouped = pd.DataFrame({"band": bands, "default": actual}).dropna(subset=["band"])
        for band, part in grouped.groupby("band", observed=True):
            if hasattr(band, "left") and hasattr(band, "right"):
                label = f"{band.left:,.0f}–{band.right:,.0f}"
            else:
                label = str(band)
            rows.append(
                {
                    "feature": column,
                    "level": label,
                    "n": int(len(part)),
                    "defaults": int(part["default"].sum()),
                    "default_rate": float(part["default"].mean()),
                }
            )
    return pd.DataFrame(rows)


def leading_segment_features(importance: pd.DataFrame) -> tuple[str, str]:
    """Strongest categorical column and strongest numeric column by permutation importance."""
    ordered = [str(name) for name in importance["feature"].tolist()]
    categorical = [name for name in ordered if name in config.CATEGORICAL_FEATURES]
    numeric = [name for name in ordered if name in config.NUMERIC_FEATURES]
    top_categorical = categorical[0] if categorical else config.CATEGORICAL_FEATURES[0]
    top_numeric = numeric[0] if numeric else config.NUMERIC_FEATURES[0]
    return top_categorical, top_numeric
