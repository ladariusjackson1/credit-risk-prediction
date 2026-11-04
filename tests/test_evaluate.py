"""Cost math, cutoff selection, and permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.evaluate import (
    baseline_costs,
    choose_threshold,
    cost_ratio_sensitivity,
    expected_cost,
    illustrative_dollars,
    permutation_importance_table,
    predict_at_threshold,
    row_at_threshold,
    score_at_threshold,
    threshold_sweep,
)
from credit_risk.features import split_features_target
from credit_risk.models import build_pipeline
from tests.synthetic import make_credit_frame


def test_statlog_cost_and_baselines():
    actual = [0, 0, 1, 1]
    predicted = [0, 1, 0, 1]
    assert expected_cost(actual, predicted) == pytest.approx(6.0)
    assert illustrative_dollars(6) == pytest.approx(1200.0)
    with pytest.raises(ValueError, match="non-negative"):
        expected_cost(actual, predicted, cost_false_approval=-1)
    with pytest.raises(ValueError, match="positive"):
        illustrative_dollars(1, denial_unit=0)

    book = np.array([1] * 30 + [0] * 70)
    baselines = baseline_costs(book)
    assert baselines["approve_all"] == pytest.approx(150)
    assert baselines["deny_all"] == pytest.approx(70)
    assert baselines["approve_all_usd"] == pytest.approx(30_000)
    assert baselines["deny_all_usd"] == pytest.approx(14_000)


def test_threshold_rules():
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    low = predict_at_threshold(scores, 0.3)
    high = predict_at_threshold(scores, 0.7)
    assert (low == 0).sum() < (high == 0).sum()
    with pytest.raises(ValueError, match="between 0 and 1"):
        predict_at_threshold(scores, 1.2)

    perfect = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    sweep = threshold_sweep(labels, perfect, thresholds=[0.2, 0.4, 0.6])
    assert choose_threshold(sweep) == pytest.approx(0.6)
    assert row_at_threshold(sweep, 0.4)["expected_cost"] == pytest.approx(0)

    with pytest.raises(ValueError, match="empty|include"):
        choose_threshold(pd.DataFrame())
    with pytest.raises(ValueError, match="not in the sweep"):
        row_at_threshold(sweep, 0.9)
    with pytest.raises(ValueError, match="same length"):
        threshold_sweep([0, 1], [0.2])
    with pytest.raises(ValueError, match="both classes"):
        score_at_threshold([0, 0], [0.2, 0.8], 0.5)

    # Deny everyone when the cutoff is 0, because every probability is at least 0.
    shut = score_at_threshold(labels, perfect, 0.0)
    assert shut["approvals"] == 0
    assert shut["default_rate_among_approved"] is None
    assert shut["expected_cost"] == pytest.approx(2.0)


def test_cost_ratio_sensitivity_returns_each_policy():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.45, 0.55, 0.8, 0.9])
    table = cost_ratio_sensitivity(labels, scores, ratios=((2.0, 1.0), (5.0, 1.0), (10.0, 1.0)))
    assert list(table["ratio"]) == [2.0, 5.0, 10.0]
    assert table["threshold"].between(0, 1).all()
    assert (table["illustrative_false_denial_usd"] == 200).all()


def test_permutation_importance_ranks_the_planted_signal():
    features, _target = split_features_target(make_credit_frame(160, seed=13))
    rng = np.random.default_rng(0)
    planted = pd.Series((rng.random(len(features)) < 0.4).astype(int), name="default")
    features = features.copy()
    features["duration"] = np.where(
        planted.to_numpy() == 1,
        rng.normal(42, 4, len(features)),
        rng.normal(10, 4, len(features)),
    )
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(features, planted)
    with pytest.raises(ValueError, match="at least 1"):
        permutation_importance_table(pipeline, features, planted, n_repeats=0)
    table = permutation_importance_table(pipeline, features, planted, n_repeats=5)
    assert table.iloc[0]["feature"] == "duration"
    assert table["importance_mean"].is_monotonic_decreasing
