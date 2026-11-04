"""Calibration points and the threshold curve the desk slider redraws."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from credit_risk.evaluate import baseline_costs, illustrative_dollars, threshold_sweep


def constant_brier(y_true) -> float:
    """Brier score of predicting this sample's own default rate for every row."""
    actual = np.asarray(y_true).astype(int)
    if len(actual) == 0:
        raise ValueError("y_true is empty.")
    base_rate = float(actual.mean())
    return float(brier_score_loss(actual, np.full(len(actual), base_rate)))


def calibration_frame(y_true, proba, n_bins: int = 8) -> pd.DataFrame:
    """Quantile-bin reliability table: mean predicted probability and observed rate."""
    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    if len(actual) != len(scores):
        raise ValueError("y_true and proba must have the same length.")
    if len(np.unique(actual)) < 2:
        raise ValueError("Calibration requires both classes.")
    bins = int(min(n_bins, max(3, len(actual) // 15)))
    observed, predicted = calibration_curve(actual, scores, n_bins=bins, strategy="quantile")
    return pd.DataFrame(
        {
            "mean_predicted": np.asarray(predicted, dtype=float),
            "observed_rate": np.asarray(observed, dtype=float),
        }
    )


def policy_curve(
    y_true,
    proba,
    cost_false_approval: float,
    cost_false_denial: float,
) -> pd.DataFrame:
    """Cost and illustrative profit versus approving everyone, at every cutoff.

    Profit here is illustrative dollars saved against the approve-everyone
    policy under the same loss ratio. It is not expected loss and not revenue.
    """
    sweep = threshold_sweep(
        y_true,
        proba,
        cost_false_approval=cost_false_approval,
        cost_false_denial=cost_false_denial,
    )
    baselines = baseline_costs(
        y_true,
        cost_false_approval=cost_false_approval,
        cost_false_denial=cost_false_denial,
    )
    denial_unit = float(cost_false_denial) if cost_false_denial else 1.0
    dollars = [
        illustrative_dollars(cost, denial_unit) for cost in sweep["expected_cost"].to_numpy()
    ]
    sweep = sweep.copy()
    sweep["illustrative_cost_usd"] = dollars
    sweep["savings_vs_approve_all_usd"] = (
        baselines["approve_all_usd"] - sweep["illustrative_cost_usd"]
    )
    sweep["savings_vs_deny_all_usd"] = baselines["deny_all_usd"] - sweep["illustrative_cost_usd"]
    sweep.attrs["baselines"] = baselines
    return sweep
