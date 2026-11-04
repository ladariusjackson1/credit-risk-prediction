"""Calibration table and the slider's profit curve."""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.curves import calibration_frame, constant_brier, policy_curve


def test_constant_brier_and_calibration_bins():
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert constant_brier(labels) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="empty"):
        constant_brier([])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    table = calibration_frame(labels, scores, n_bins=3)
    assert len(table) >= 1
    assert table["mean_predicted"].between(0, 1).all()
    assert table["observed_rate"].between(0, 1).all()
    with pytest.raises(ValueError, match="same length"):
        calibration_frame(labels, scores[:3])
    with pytest.raises(ValueError, match="both classes"):
        calibration_frame(np.zeros(6), scores)


def test_profit_is_savings_versus_approving_everyone():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    curve = policy_curve(labels, scores, cost_false_approval=5, cost_false_denial=1)
    # Approving everyone costs 2 false approvals = 10 units = $2,000.
    # Cutoff 1.0 approves scores below 1, which is everyone here, cost $2,000, savings 0.
    at_one = curve.loc[np.isclose(curve["threshold"], 1.0)].iloc[0]
    assert at_one["illustrative_cost_usd"] == pytest.approx(2000)
    assert at_one["savings_vs_approve_all_usd"] == pytest.approx(0)
    # Cutoff 0.5 denies the two defaulters and approves the two payers. Cost 0.
    at_half = curve.loc[np.isclose(curve["threshold"], 0.5)].iloc[0]
    assert at_half["expected_cost"] == pytest.approx(0)
    assert at_half["savings_vs_approve_all_usd"] == pytest.approx(2000)
    assert at_half["savings_vs_deny_all_usd"] == pytest.approx(400)
