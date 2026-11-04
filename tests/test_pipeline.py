"""End-to-end study on a synthetic book, plus JSON conversion."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from credit_risk import config
from credit_risk.pipeline import main, run_analysis, to_builtin
from tests.synthetic import make_credit_frame


@pytest.fixture(autouse=True)
def _small_models(monkeypatch):
    monkeypatch.setattr(config, "RF_N_ESTIMATORS", 15)
    monkeypatch.setattr(config, "GB_MAX_ITER", 15)
    monkeypatch.setattr(config, "GB_MIN_SAMPLES_LEAF", 5)


def test_to_builtin_handles_numpy_and_missing():
    sentinel = object()
    payload = to_builtin(
        {
            "flag": np.bool_(True),
            "count": np.int64(3),
            "score": np.float64(0.5),
            "gap": float("nan"),
            "huge": float("inf"),
            "items": (np.int64(1), None),
            "blank": pd.NA,
            "label": "good",
            "vector": np.array([1, 2]),
            "sentinel": sentinel,
        }
    )
    assert payload["flag"] is True
    assert payload["count"] == 3
    assert payload["score"] == pytest.approx(0.5)
    assert payload["gap"] is None
    assert payload["huge"] is None
    assert payload["items"] == [1, None]
    assert payload["blank"] is None
    assert payload["label"] == "good"
    assert payload["vector"] == [1, 2]
    assert payload["sentinel"] is sentinel
    assert math.isfinite(payload["score"])


def test_run_analysis_writes_the_study(tmp_path):
    payload = run_analysis(
        make_credit_frame(150, seed=21),
        reports_dir=tmp_path / "reports",
        model_path=tmp_path / "models" / "credit_risk_model.joblib",
        n_splits=3,
        n_importance_repeats=2,
    )
    assert payload["selected_model"] in config.MODEL_NAMES
    assert 0.0 <= payload["selected_threshold"] <= 1.0
    assert (tmp_path / "reports" / "metrics.json").exists()
    assert (tmp_path / "reports" / "test_metrics.csv").exists()
    assert (tmp_path / "reports" / "figures" / "roc_curves.png").stat().st_size > 0
    assert (tmp_path / "reports" / "figures" / "permutation_importance.png").stat().st_size > 0
    assert (tmp_path / "models" / "credit_risk_model.joblib").exists()
    assert payload["dataset"]["n"] == 150
    assert payload["fairness"]["dropped_features"] == ["personal_status", "foreign_worker"]
    assert len(payload["cost_sensitivity"]) == 3


def test_main_cli(tmp_path, monkeypatch):
    frame = make_credit_frame(120, seed=22)
    cache = tmp_path / "german_credit.csv"
    frame.to_csv(cache, index=False)
    monkeypatch.setattr("credit_risk.pipeline.load_german_credit", lambda **kwargs: frame)
    payload = main(
        [
            "--cache",
            str(cache),
            "--reports",
            str(tmp_path / "reports"),
            "--model",
            str(tmp_path / "model.joblib"),
            "--splits",
            "3",
            "--importance-repeats",
            "2",
        ]
    )
    assert payload["design"]["n_splits"] == 3
    assert (tmp_path / "model.joblib").exists()
