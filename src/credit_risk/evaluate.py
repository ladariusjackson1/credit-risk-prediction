"""Decision metrics, the Statlog cost function, and permutation importance."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from credit_risk import config


def illustrative_dollars(
    relative_units: float, denial_unit: float = config.COST_FALSE_DENIAL
) -> float:
    """Convert cost-matrix units into illustrative dollars.

    One unit of false-denial cost is ``ILLUSTRATIVE_DENIAL_USD`` dollars.
    ``denial_unit`` is the false-denial weight used to build ``relative_units``,
    which is 1 in the published matrix.
    """
    if denial_unit <= 0:
        raise ValueError("denial_unit must be positive.")
    return float(relative_units) * (config.ILLUSTRATIVE_DENIAL_USD / float(denial_unit))


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Counts with the positive class fixed as default (1)."""
    matrix = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    return {
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "tp": int(true_positive),
    }


def predict_at_threshold(proba, threshold: float) -> np.ndarray:
    """Deny (predict default) when the score is at least ``threshold``."""
    cutoff = float(threshold)
    if cutoff < 0.0 or cutoff > 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    return (np.asarray(proba, dtype=float) >= cutoff).astype(int)


def expected_cost(
    y_true,
    y_pred,
    cost_false_approval: float = config.COST_FALSE_APPROVAL,
    cost_false_denial: float = config.COST_FALSE_DENIAL,
) -> float:
    """Statlog cost. A false approval is an actual default that was booked."""
    if cost_false_approval < 0 or cost_false_denial < 0:
        raise ValueError("Costs must be non-negative.")
    counts = confusion_counts(y_true, y_pred)
    # fn: actual bad, predicted good. fp: actual good, predicted bad.
    return float(counts["fn"] * cost_false_approval + counts["fp"] * cost_false_denial)


def _aligned(y_true, proba) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    if len(actual) != len(scores):
        raise ValueError("y_true and proba must have the same length.")
    return actual, scores


def threshold_sweep(
    y_true,
    proba,
    thresholds: Sequence[float] | None = None,
    cost_false_approval: float = config.COST_FALSE_APPROVAL,
    cost_false_denial: float = config.COST_FALSE_DENIAL,
) -> pd.DataFrame:
    """Score every cutoff from 'deny everyone' (0) to 'approve almost everyone' (1)."""
    actual, scores = _aligned(y_true, proba)
    if thresholds is None:
        grid = np.unique(np.round(np.linspace(0.0, 1.0, 101), 2))
    else:
        grid = np.asarray(list(thresholds), dtype=float)
    rows = []
    n = len(actual)
    for cutoff in grid:
        decision = predict_at_threshold(scores, float(cutoff))
        counts = confusion_counts(actual, decision)
        predicted_positive = counts["tp"] + counts["fp"]
        actual_positive = counts["tp"] + counts["fn"]
        precision = counts["tp"] / predicted_positive if predicted_positive else 0.0
        recall = counts["tp"] / actual_positive if actual_positive else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        approvals = counts["tn"] + counts["fn"]
        cost = counts["fn"] * cost_false_approval + counts["fp"] * cost_false_denial
        rows.append(
            {
                "threshold": float(cutoff),
                "precision_bad": precision,
                "recall_bad": recall,
                "f1_bad": f1,
                "false_approvals": counts["fn"],
                "false_denials": counts["fp"],
                "true_denials": counts["tp"],
                "true_approvals": counts["tn"],
                "approval_rate": approvals / n,
                "default_rate_among_approved": (counts["fn"] / approvals) if approvals else np.nan,
                "expected_cost": float(cost),
                "expected_cost_per_applicant": float(cost) / n,
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame) -> float:
    """Lowest expected cost. Equal costs keep the higher cutoff (more loans booked)."""
    if sweep.empty or "expected_cost" not in sweep or "threshold" not in sweep:
        raise ValueError("Sweep must include threshold and expected_cost.")
    best = float(sweep["expected_cost"].min())
    candidates = sweep.loc[np.isclose(sweep["expected_cost"].to_numpy(), best)]
    return float(candidates["threshold"].max())


def row_at_threshold(sweep: pd.DataFrame, threshold: float) -> pd.Series:
    match = sweep.loc[np.isclose(sweep["threshold"].to_numpy(), float(threshold))]
    if match.empty:
        raise ValueError(f"Threshold {threshold} is not in the sweep.")
    return match.iloc[0]


def score_at_threshold(
    y_true,
    proba,
    threshold: float,
    cost_false_approval: float = config.COST_FALSE_APPROVAL,
    cost_false_denial: float = config.COST_FALSE_DENIAL,
) -> dict:
    """Ranking metrics plus the operating point at one cutoff."""
    actual, scores = _aligned(y_true, proba)
    if len(np.unique(actual)) < 2:
        raise ValueError("Metrics require both classes in y_true.")
    decision = predict_at_threshold(scores, threshold)
    counts = confusion_counts(actual, decision)
    approvals = counts["tn"] + counts["fn"]
    cost = expected_cost(actual, decision, cost_false_approval, cost_false_denial)
    n = len(actual)
    dollars = illustrative_dollars(cost, cost_false_denial if cost_false_denial else 1.0)
    return {
        "roc_auc": float(roc_auc_score(actual, scores)),
        "average_precision": float(average_precision_score(actual, scores)),
        "brier": float(brier_score_loss(actual, scores)),
        "threshold": float(threshold),
        "precision_bad": float(precision_score(actual, decision, zero_division=0)),
        "recall_bad": float(recall_score(actual, decision, zero_division=0)),
        "f1_bad": float(f1_score(actual, decision, zero_division=0)),
        "true_approvals": counts["tn"],
        "false_denials": counts["fp"],
        "false_approvals": counts["fn"],
        "true_denials": counts["tp"],
        "approvals": approvals,
        "denials": counts["tp"] + counts["fp"],
        "approval_rate": approvals / n,
        "default_rate_among_approved": (counts["fn"] / approvals) if approvals else None,
        "expected_cost": cost,
        "expected_cost_per_applicant": cost / n,
        "illustrative_cost_usd": dollars,
        "illustrative_cost_per_applicant_usd": dollars / n,
    }


def baseline_costs(
    y_true,
    cost_false_approval: float = config.COST_FALSE_APPROVAL,
    cost_false_denial: float = config.COST_FALSE_DENIAL,
) -> dict:
    """Cost of booking every applicant, and of booking nobody."""
    actual = np.asarray(y_true).astype(int)
    n_bad = int(actual.sum())
    n_good = int(len(actual) - n_bad)
    n = len(actual)
    approve_all = float(n_bad * cost_false_approval)
    deny_all = float(n_good * cost_false_denial)
    return {
        "n": n,
        "n_bad": n_bad,
        "n_good": n_good,
        "approve_all": approve_all,
        "deny_all": deny_all,
        "approve_all_per_applicant": approve_all / n,
        "deny_all_per_applicant": deny_all / n,
        "approve_all_usd": illustrative_dollars(approve_all, cost_false_denial or 1.0),
        "deny_all_usd": illustrative_dollars(deny_all, cost_false_denial or 1.0),
    }


def cost_ratio_sensitivity(
    y_true,
    proba,
    ratios: Sequence[tuple[float, float]] = config.COST_RATIOS,
) -> pd.DataFrame:
    """How the cutoff moves when management changes the loss ratio."""
    rows = []
    for false_approval, false_denial in ratios:
        sweep = threshold_sweep(
            y_true,
            proba,
            cost_false_approval=false_approval,
            cost_false_denial=false_denial,
        )
        cutoff = choose_threshold(sweep)
        picked = row_at_threshold(sweep, cutoff)
        per_applicant = float(picked["expected_cost_per_applicant"])
        rows.append(
            {
                "cost_false_approval": float(false_approval),
                "cost_false_denial": float(false_denial),
                "ratio": float(false_approval) / float(false_denial),
                "threshold": cutoff,
                "approval_rate": float(picked["approval_rate"]),
                "recall_bad": float(picked["recall_bad"]),
                "precision_bad": float(picked["precision_bad"]),
                "expected_cost_per_applicant": per_applicant,
                "illustrative_cost_per_applicant_usd": illustrative_dollars(
                    per_applicant, false_denial
                ),
                "illustrative_false_approval_usd": illustrative_dollars(
                    false_approval, false_denial
                ),
                "illustrative_false_denial_usd": illustrative_dollars(false_denial, false_denial),
            }
        )
    return pd.DataFrame(rows)


def permutation_importance_table(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    n_repeats: int = config.N_IMPORTANCE_REPEATS,
    random_state: int = config.RANDOM_STATE,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """Drop in ROC-AUC when each original column is shuffled.

    Importance is computed on the raw columns, through the fitted pipeline, so
    the table says ``checking_status`` rather than a one-hot piece of it.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    result = permutation_importance(
        pipeline,
        features,
        target,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": list(features.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)
