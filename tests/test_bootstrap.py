"""Bootstrap intervals on fixed scores."""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.bootstrap import auc_difference_sentence, bootstrap_scores


def test_perfect_scores_have_a_degenerate_interval():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = {"model": np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])}
    table = bootstrap_scores(labels, scores, n_boot=30, random_state=1, thresholds={"model": 0.5})
    auc = table.loc[table["metric"] == "roc_auc"].iloc[0]
    assert auc["estimate"] == pytest.approx(1.0)
    assert auc["ci_low"] == pytest.approx(1.0)
    assert auc["ci_high"] == pytest.approx(1.0)
    cost = table.loc[table["metric"] == "cost_per_applicant"].iloc[0]
    assert cost["estimate"] == pytest.approx(0.0)


def test_difference_against_an_identical_baseline_is_zero():
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6, 0.4, 0.9])
    table = bootstrap_scores(
        labels,
        {"candidate": scores, "baseline": scores.copy()},
        n_boot=40,
        random_state=2,
        baseline="baseline",
        thresholds={"candidate": 0.5, "baseline": 0.5},
    )
    gap = table.loc[(table["comparison"] == "difference") & (table["metric"] == "roc_auc")].iloc[0]
    assert gap["estimate"] == pytest.approx(0.0)
    assert gap["ci_low"] == pytest.approx(0.0)
    assert gap["ci_high"] == pytest.approx(0.0)
    assert bool(gap["includes_zero"]) is True
    first = bootstrap_scores(labels, {"candidate": scores}, n_boot=25, random_state=7)
    second = bootstrap_scores(labels, {"candidate": scores}, n_boot=25, random_state=7)
    assert first["ci_low"].tolist() == second["ci_low"].tolist()


def test_bootstrap_rejects_bad_requests():
    labels = np.array([0, 1, 0, 1])
    scores = {"model": np.array([0.2, 0.8, 0.3, 0.7])}
    with pytest.raises(ValueError, match="at least 20"):
        bootstrap_scores(labels, scores, n_boot=5)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_scores(labels, scores, n_boot=20, alpha=1)
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_scores(
            np.array([0, 0, 0, 0]), {"model": np.array([0.1, 0.2, 0.3, 0.4])}, n_boot=20
        )
    with pytest.raises(ValueError, match="do not match"):
        bootstrap_scores(labels, {"model": np.array([0.2, 0.8])}, n_boot=20)
    with pytest.raises(ValueError, match="empty"):
        bootstrap_scores(labels, {}, n_boot=20)
    with pytest.raises(ValueError, match="not in the scores"):
        bootstrap_scores(labels, scores, n_boot=20, baseline="missing")
    with pytest.raises(ValueError, match="Missing thresholds"):
        bootstrap_scores(labels, scores, n_boot=20, thresholds={})


def test_auc_sentence_states_whether_zero_is_inside():
    overlap = auc_difference_sentence("Boosting", "Logistic", 0.01, -0.02, 0.04)
    assert "includes zero" in overlap
    assert "+0.010" in overlap
    separated = auc_difference_sentence("Boosting", "Logistic", 0.05, 0.01, 0.09)
    assert "excludes zero" in separated
    assert "higher" in separated
    lower = auc_difference_sentence("Boosting", "Logistic", -0.04, -0.08, -0.01)
    assert "lower" in lower
