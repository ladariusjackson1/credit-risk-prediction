"""Stratified cross-validation used to compare models without the holdout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from credit_risk import config
from credit_risk.models import build_pipeline, positive_class_probability


@dataclass(frozen=True)
class ComparisonResult:
    """Fold scores, a one-row-per-model summary, and out-of-fold default scores."""

    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    best_model: str
    oof_proba: dict[str, np.ndarray]


def select_best_model(summary: pd.DataFrame) -> str:
    """Highest mean ROC-AUC, then tighter fold spread, then the model name."""
    if summary.empty:
        raise ValueError("Model summary is empty.")
    ranked = summary.sort_values(
        by=["roc_auc_mean", "roc_auc_std", "model"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return str(ranked.iloc[0]["model"])


def _check_split_request(y: pd.Series, n_splits: int) -> None:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    counts = y.value_counts()
    smallest = int(counts.min()) if len(counts) else 0
    if set(counts.index.tolist()) != {0, 1} or smallest < n_splits:
        raise ValueError(
            f"Need both classes and at least {n_splits} rows in each. "
            f"Smallest class has {smallest} rows."
        )


def cross_validate_models(
    features: pd.DataFrame,
    target: pd.Series,
    model_names: Sequence[str] = config.MODEL_NAMES,
    n_splits: int = config.N_SPLITS,
    random_state: int = config.RANDOM_STATE,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> ComparisonResult:
    """Fit each model out of fold and record ROC-AUC and average precision.

    Out-of-fold scores line up with ``target`` by position. They are the scores
    used to choose a cutoff, so the holdout is not involved in that choice.
    """
    _check_split_request(target, n_splits)
    names = tuple(model_names)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    observed = target.to_numpy()
    oof = {name: np.zeros(len(target), dtype=float) for name in names}
    rows: list[dict[str, float | int | str]] = []

    for name in names:
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(features, target), start=1):
            pipeline = build_pipeline(name, numeric_features, categorical_features)
            pipeline.fit(features.iloc[train_idx], target.iloc[train_idx])
            scores = positive_class_probability(pipeline, features.iloc[valid_idx])
            oof[name][valid_idx] = scores
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "roc_auc": float(roc_auc_score(observed[valid_idx], scores)),
                    "average_precision": float(
                        average_precision_score(observed[valid_idx], scores)
                    ),
                }
            )

    fold_metrics = pd.DataFrame(rows)
    summary = fold_metrics.groupby("model", as_index=False).agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_std=("average_precision", "std"),
    )
    summary["model"] = pd.Categorical(summary["model"], categories=list(names), ordered=True)
    summary = summary.sort_values("model").reset_index(drop=True)
    summary["model"] = summary["model"].astype(str)
    summary["display_name"] = summary["model"].map(config.DISPLAY_NAMES)
    return ComparisonResult(
        fold_metrics=fold_metrics,
        summary=summary,
        best_model=select_best_model(summary),
        oof_proba=oof,
    )
