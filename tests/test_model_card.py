"""The model card is filled from the payload, not from remembered numbers."""

from __future__ import annotations

import pytest

from credit_risk.model_card import render_model_card


def _payload() -> dict:
    row = {
        "model": "gradient_boosting",
        "display_name": "Gradient boosting",
        "role": "Selected model",
        "roc_auc_cv_mean": 0.812345,
        "roc_auc_cv_std": 0.021,
        "oof_roc_auc": 0.801,
        "oof_roc_auc_ci_low": 0.761,
        "oof_roc_auc_ci_high": 0.841,
        "oof_brier": 0.171,
        "oof_brier_ci_low": 0.151,
        "oof_brier_ci_high": 0.191,
        "threshold": 0.15,
        "holdout_roc_auc": 0.771,
        "holdout_brier": 0.187654,
        "holdout_illustrative_cost_usd": 22800,
    }
    return {
        "dataset": {
            "name": "Statlog sample",
            "citation": "Hofmann 1994",
            "n": 1000,
            "n_train": 800,
            "n_test": 200,
            "default_rate": 0.3,
            "missing_numeric": 0,
            "missing_categorical": 0,
        },
        "policy": {
            "positive_class": "default",
            "cost_false_approval": 5,
            "cost_false_denial": 1,
            "illustrative_denial_usd": 200,
            "illustrative_false_approval_usd": 1000,
            "selection_rule": "Highest training-fold ROC-AUC.",
            "threshold_rule": "Lowest out-of-fold cost.",
        },
        "selected_model": "gradient_boosting",
        "selected_threshold": 0.15,
        "tuned_baseline": {
            "display_name": "Tuned logistic baseline",
            "C": 1.0,
            "C_grid": [0.1, 1.0, 10.0],
            "selection_rule": "Pick C on the training folds.",
            "optimism_note": "The same folds chose C.",
        },
        "best_vs_baseline": {
            "sentence": "DISTINCT_SENTENCE_9f3a.",
            "oof_brier_difference": -0.012345,
            "oof_brier_ci_low": -0.03,
            "oof_brier_ci_high": 0.01,
            "oof_cost_per_applicant_difference": -0.02,
            "oof_cost_ci_low": -0.05,
            "oof_cost_ci_high": 0.01,
            "holdout_cost_usd_model": 22800,
            "holdout_cost_usd_baseline": 22000,
        },
        "bootstrap": {
            "method": "Stratified percentile bootstrap.",
            "n_boot": 1000,
            "alpha": 0.05,
            "random_state": 42,
        },
        "comparison": [row],
        "group_fairness": {
            "age_cut": 25,
            "sex_definition": "first word of personal_status.",
            "oof": {"note": "OOF_NOTE_age_gap."},
            "holdout": {"note": "HOLDOUT_NOTE_age_gap."},
        },
        "fairness": {
            "dropped_features": ["personal_status", "foreign_worker"],
            "cv_roc_auc_with_protected": 0.793,
            "cv_roc_auc_without_protected": 0.786,
            "mean_auc_change_when_dropped": -0.007,
        },
        "calibration": {
            "holdout_brier": 0.187654,
            "holdout_brier_constant": 0.21,
            "oof_brier": 0.171,
            "oof_brier_constant": 0.21,
        },
        "top_features": [
            {"feature": "checking_status", "importance_mean": 0.0826, "importance_std": 0.0212}
        ],
        "design": {"random_state": 42, "test_size": 0.2, "n_splits": 5},
        "environment": {
            "python": "3.12.3",
            "scikit_learn": "1.9.1",
            "pandas": "3.1.0",
            "numpy": "2.5.4",
        },
    }


def test_card_uses_the_payload_and_states_the_limits():
    text = render_model_card(_payload())
    assert "Intended use" in text
    assert "Out of scope" in text
    assert "DISTINCT_SENTENCE_9f3a." in text
    assert "OOF_NOTE_age_gap." in text
    assert "HOLDOUT_NOTE_age_gap." in text
    assert "0.188" in text
    assert "0.812" in text
    assert "-0.012" in text
    assert "checking_status" in text
    assert "not a fair-lending" in text or "Fair-lending compliance" in text
    assert "live lending" in text


def test_card_formats_a_missing_number_as_not_estimated():
    payload = _payload()
    payload["dataset"]["default_rate"] = None
    payload["policy"]["illustrative_denial_usd"] = None
    payload["comparison"][0]["roc_auc_cv_mean"] = None
    payload["comparison"][0]["oof_roc_auc_ci_low"] = None
    text = render_model_card(payload)
    assert text.count("not estimated") >= 3


def test_card_refuses_a_partial_payload():
    payload = _payload()
    del payload["calibration"]
    with pytest.raises(KeyError, match="calibration"):
        render_model_card(payload)


def test_card_without_importance_says_so():
    payload = _payload()
    payload["top_features"] = []
    text = render_model_card(payload)
    assert "No permutation-importance row was stored." in text
