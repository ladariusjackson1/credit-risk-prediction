"""Shared fixtures. The logistic fit is done once per session."""

from __future__ import annotations

import pytest

from credit_risk.config import (
    ARTIFACT_VERSION,
    CATEGORICAL_FEATURES,
    COST_FALSE_APPROVAL,
    COST_FALSE_DENIAL,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)
from credit_risk.features import category_codebook, reference_applicant, split_features_target
from credit_risk.models import build_pipeline
from tests.synthetic import make_credit_frame


@pytest.fixture(scope="module")
def logistic_artifact() -> dict:
    frame = make_credit_frame(120, seed=3)
    features, target = split_features_target(frame)
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(features, target)
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": "logistic_regression",
        "display_name": "Logistic regression",
        "pipeline": pipeline,
        "threshold": 0.40,
        "review_band": 0.05,
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_columns": list(NUMERIC_FEATURES),
        "categorical_columns": list(CATEGORICAL_FEATURES),
        "category_levels": category_codebook(features),
        "reference_applicant": reference_applicant(features),
        "cost_false_approval": COST_FALSE_APPROVAL,
        "cost_false_denial": COST_FALSE_DENIAL,
        "illustrative_false_approval_usd": 1000.0,
        "illustrative_denial_usd": 200.0,
    }
