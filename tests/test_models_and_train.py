"""Estimators and the cross-validation comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk import config
from credit_risk.features import split_features_target
from credit_risk.models import build_pipeline, make_classifier, positive_class_probability
from credit_risk.train import (
    choose_logistic_c,
    cross_validate_models,
    select_best_model,
    select_logistic_c,
)
from tests.synthetic import make_credit_frame


@pytest.fixture(autouse=True)
def _small_models(monkeypatch):
    monkeypatch.setattr(config, "RF_N_ESTIMATORS", 20)
    monkeypatch.setattr(config, "GB_MAX_ITER", 20)
    monkeypatch.setattr(config, "GB_MIN_SAMPLES_LEAF", 5)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        make_classifier("neural_net")


def test_each_model_emits_default_probabilities():
    features, target = split_features_target(make_credit_frame(80, seed=10))
    for name in config.MODEL_NAMES:
        pipeline = build_pipeline(name)
        pipeline.fit(features, target)
        scores = positive_class_probability(pipeline, features)
        assert scores.shape == (len(features),)
        assert np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()


def test_positive_class_must_be_present():
    class _OnlyGoods:
        classes_ = np.array([0])

        def predict_proba(self, features):
            return np.zeros((len(features), 1))

    class _Pipeline:
        named_steps = {"model": _OnlyGoods()}

        def predict_proba(self, features):
            return self.named_steps["model"].predict_proba(features)

    features, _target = split_features_target(make_credit_frame(20, seed=11))
    with pytest.raises(RuntimeError, match="positive class"):
        positive_class_probability(_Pipeline(), features.iloc[:3])


def test_select_best_tie_breaks():
    with pytest.raises(ValueError, match="empty"):
        select_best_model(pd.DataFrame())
    summary = pd.DataFrame(
        [
            {"model": "random_forest", "roc_auc_mean": 0.80, "roc_auc_std": 0.02},
            {"model": "gradient_boosting", "roc_auc_mean": 0.80, "roc_auc_std": 0.02},
            {"model": "logistic_regression", "roc_auc_mean": 0.79, "roc_auc_std": 0.01},
        ]
    )
    assert select_best_model(summary) == "gradient_boosting"
    summary.loc[summary["model"] == "random_forest", "roc_auc_std"] = 0.01
    assert select_best_model(summary) == "random_forest"


def test_cross_validation_records_oof_scores():
    features, target = split_features_target(make_credit_frame(90, seed=12))
    with pytest.raises(ValueError, match="at least 2"):
        cross_validate_models(features, target, n_splits=1)
    too_small = target.copy()
    too_small.iloc[:] = 0
    too_small.iloc[:3] = 1
    with pytest.raises(ValueError, match="both classes"):
        cross_validate_models(features, too_small, n_splits=5)

    result = cross_validate_models(features, target, n_splits=3)
    assert set(result.summary["model"]) == set(config.MODEL_NAMES)
    assert result.best_model in config.MODEL_NAMES
    assert len(result.fold_metrics) == 9
    assert result.summary["roc_auc_std"].notna().all()
    for name, scores in result.oof_proba.items():
        assert len(scores) == len(target)
        assert np.isfinite(scores).all()
        assert name in config.MODEL_NAMES


def test_logistic_c_is_chosen_on_training_folds_only():
    with pytest.raises(ValueError, match="logistic_c applies"):
        make_classifier("random_forest", logistic_c=0.1)
    with pytest.raises(ValueError, match="positive finite"):
        make_classifier("logistic_regression", logistic_c=0)
    with pytest.raises(ValueError, match="roc_auc_mean"):
        choose_logistic_c(pd.DataFrame())
    tied = pd.DataFrame({"C": [0.1, 1.0, 10.0], "roc_auc_mean": [0.70, 0.70, 0.70]})
    assert choose_logistic_c(tied) == pytest.approx(1.0)
    farther = pd.DataFrame({"C": [0.1, 10.0], "roc_auc_mean": [0.80, 0.80]})
    assert choose_logistic_c(farther) == pytest.approx(0.1)
    winner = pd.DataFrame({"C": [0.1, 1.0, 10.0], "roc_auc_mean": [0.61, 0.62, 0.90]})
    assert choose_logistic_c(winner) == pytest.approx(10.0)

    features, target = split_features_target(make_credit_frame(90, seed=13))
    with pytest.raises(ValueError, match="empty"):
        select_logistic_c(features, target, n_splits=3, grid=())
    chosen, table, scores = select_logistic_c(features, target, n_splits=3, grid=(0.1, 1.0))
    assert chosen in (0.1, 1.0)
    assert set(table["C"]) == {0.1, 1.0}
    assert len(scores) == len(target)
    assert np.isfinite(scores).all()
