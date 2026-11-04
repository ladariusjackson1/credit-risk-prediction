"""Preprocessing, codebook order, and the feature/target split."""

from __future__ import annotations

import numpy as np

from credit_risk.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from credit_risk.features import (
    build_preprocessor,
    feature_lists,
    ordered_levels,
    split_features_target,
)
from credit_risk.models import build_pipeline, positive_class_probability
from tests.synthetic import make_credit_frame


def test_ordered_levels_and_dropped_columns():
    levels = ordered_levels("checking_status", ["no checking", "<0", "brand new"])
    assert levels == ["<0", "no checking", "brand new"]
    numeric, categorical = feature_lists(["personal_status", "foreign_worker", "age"])
    assert "age" not in numeric
    assert "duration" in numeric
    assert "personal_status" not in categorical
    assert "checking_status" in categorical


def test_split_and_preprocessor_shape():
    raw = make_credit_frame(40, seed=8)
    features, target = split_features_target(raw)
    assert list(features.columns) == FEATURE_COLUMNS
    assert target.isin([0, 1]).all()
    assert int(target.sum()) == int(raw["class"].eq("bad").sum())
    transformed = build_preprocessor().fit_transform(features, target)
    assert transformed.shape[0] == len(features)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
    assert np.isfinite(transformed).all()


def test_pipeline_imputes_gaps():
    features, target = split_features_target(make_credit_frame(50, seed=9))
    features = features.copy()
    features.loc[0, "age"] = np.nan
    features.loc[0, "purpose"] = np.nan
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(features, target)
    scores = positive_class_probability(pipeline, features.iloc[:4])
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert set(CATEGORICAL_FEATURES).issubset(features.columns)
