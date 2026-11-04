"""Chart writers create non-empty image files."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.plots import (
    save_calibration,
    save_confusion_matrices,
    save_importance,
    save_model_comparison,
    save_policy_costs,
    save_precision_recall_curves,
    save_roc_curves,
    save_score_distribution,
    save_segment_rates,
    save_threshold_cost,
)


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "roc_auc_mean": 0.74,
                "roc_auc_std": 0.03,
                "average_precision_mean": 0.55,
                "average_precision_std": 0.04,
            },
            {
                "model": "random_forest",
                "roc_auc_mean": 0.77,
                "roc_auc_std": 0.02,
                "average_precision_mean": 0.58,
                "average_precision_std": 0.03,
            },
            {
                "model": "gradient_boosting",
                "roc_auc_mean": 0.79,
                "roc_auc_std": 0.02,
                "average_precision_mean": 0.60,
                "average_precision_std": 0.03,
            },
        ]
    )


def test_every_chart_is_written(tmp_path):
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    scores = {
        "logistic_regression": np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.35, 0.65]),
        "random_forest": np.array([0.15, 0.25, 0.45, 0.55, 0.75, 0.85, 0.3, 0.7]),
        "gradient_boosting": np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.25, 0.72]),
    }
    sweep = pd.DataFrame({"threshold": [0.0, 0.4, 1.0], "expected_cost": [30.0, 12.0, 40.0]})
    baselines = {
        "approve_all": 40.0,
        "deny_all": 30.0,
        "approve_all_usd": 8000.0,
        "deny_all_usd": 6000.0,
    }
    test_table = pd.DataFrame(
        [
            {"model": "logistic_regression", "illustrative_cost_usd": 5000.0},
            {"model": "random_forest", "illustrative_cost_usd": 4200.0},
            {"model": "gradient_boosting", "illustrative_cost_usd": 3900.0},
        ]
    )
    importance = pd.DataFrame(
        {
            "feature": ["checking_status", "duration", "age"],
            "importance_mean": [0.08, 0.05, 0.01],
            "importance_std": [0.01, 0.01, 0.005],
        }
    )
    categories = pd.DataFrame(
        {
            "feature": ["checking_status", "checking_status"],
            "level": ["<0", "no checking"],
            "n": [20, 30],
            "default_rate": [0.45, 0.12],
        }
    )
    numeric = pd.DataFrame(
        {
            "feature": ["other"],
            "level": ["unused"],
            "n": [10],
            "default_rate": [0.2],
        }
    )
    paths = [
        save_model_comparison(_summary(), tmp_path / "comparison.png"),
        save_roc_curves(labels, scores, tmp_path / "roc.png"),
        save_precision_recall_curves(labels, scores, tmp_path / "pr.png"),
        save_confusion_matrices(labels, scores["gradient_boosting"], 0.4, tmp_path / "cm.png"),
        save_threshold_cost(sweep, 0.4, baselines, tmp_path / "cost.png"),
        save_policy_costs(test_table, baselines, "gradient_boosting", tmp_path / "policy.png"),
        save_importance(importance, tmp_path / "importance.png"),
        save_score_distribution(labels, scores["gradient_boosting"], 0.4, tmp_path / "scores.png"),
        save_calibration(labels, scores["gradient_boosting"], tmp_path / "calibration.png"),
        save_segment_rates(
            categories, numeric, "checking_status", "duration", 0.3, tmp_path / "segments.png"
        ),
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
