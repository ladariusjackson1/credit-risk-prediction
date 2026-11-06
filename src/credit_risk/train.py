"""Stratified cross-validation used to compare models without the holdout."""

from __future__ import annotations

import math
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


LOGISTIC_SELECTION_RULE = (
    "On the training set only, C is chosen from the fixed grid 0.1, 1, and 10 "
    "by mean stratified cross-validated ROC-AUC. Ties prefer the C closest to 1, "
    "then the smaller C."
)
LOGISTIC_OPTIMISM_NOTE = (
    "C is chosen on the same folds that are reported, so the baseline's "
    "cross-validated AUC can look slightly better than a nested estimate. "
    "The holdout is not used to choose C."
)


def choose_logistic_c(grid_results: pd.DataFrame) -> float:
    """Highest mean ROC-AUC. Ties prefer the C closest to 1, then the smaller C."""
    required = {"C", "roc_auc_mean"}
    if grid_results.empty or not required <= set(grid_results.columns):
        raise ValueError("Grid results must include C and roc_auc_mean.")
    table = grid_results.copy()
    table["tie_distance"] = (table["C"].astype(float) - 1.0).abs()
    ranked = table.sort_values(
        by=["roc_auc_mean", "tie_distance", "C"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["C"])


def select_logistic_c(
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = config.N_SPLITS,
    random_state: int = config.RANDOM_STATE,
    grid: Sequence[float] = config.LOGISTIC_C_GRID,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> tuple[float, pd.DataFrame, np.ndarray]:
    """Choose logistic ``C`` on training folds and return that model's out-of-fold scores.

    The same folds score every C. The reported cross-validated AUC is therefore
    a little optimistic relative to a nested search: C was picked on the folds
    being measured. The holdout is not involved.
    """
    if len(tuple(grid)) == 0:
        raise ValueError("The logistic C grid is empty.")
    rows: list[dict[str, float]] = []
    oof_by_c: dict[float, np.ndarray] = {}
    for penalty in grid:
        result = cross_validate_models(
            features,
            target,
            model_names=("logistic_regression",),
            n_splits=n_splits,
            random_state=random_state,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            logistic_c=float(penalty),
        )
        summary = result.summary.iloc[0]
        chosen_key = float(penalty)
        rows.append(
            {
                "C": chosen_key,
                "roc_auc_mean": float(summary["roc_auc_mean"]),
                "roc_auc_std": float(summary["roc_auc_std"]),
                "average_precision_mean": float(summary["average_precision_mean"]),
                "average_precision_std": float(summary["average_precision_std"]),
            }
        )
        oof_by_c[chosen_key] = result.oof_proba["logistic_regression"]
    table = pd.DataFrame(rows)
    chosen = choose_logistic_c(table)
    scores = None
    for key, values in oof_by_c.items():
        if math.isclose(key, chosen, rel_tol=0.0, abs_tol=1e-12):
            scores = values
            break
    if scores is None:
        raise KeyError(f"No out-of-fold scores were stored for C={chosen}.")
    return chosen, table, scores


def cross_validate_models(
    features: pd.DataFrame,
    target: pd.Series,
    model_names: Sequence[str] = config.MODEL_NAMES,
    n_splits: int = config.N_SPLITS,
    random_state: int = config.RANDOM_STATE,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    logistic_c: float | None = None,
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
            pipeline = build_pipeline(
                name, numeric_features, categorical_features, logistic_c=logistic_c
            )
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
