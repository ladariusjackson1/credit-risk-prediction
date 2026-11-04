"""Segment default rates."""

from __future__ import annotations

import pandas as pd

from credit_risk.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_risk.features import split_features_target
from credit_risk.profile import (
    categorical_default_rates,
    leading_segment_features,
    numeric_band_default_rates,
)
from tests.synthetic import make_credit_frame


def test_segment_rates_cover_the_book():
    features, target = split_features_target(make_credit_frame(80, seed=14))
    categories = categorical_default_rates(features, target)
    assert set(categories["feature"]) == set(CATEGORICAL_FEATURES)
    for _feature, part in categories.groupby("feature"):
        assert int(part["n"].sum()) == len(features)
        assert part["default_rate"].between(0, 1).all()

    bands = numeric_band_default_rates(features, target, q=4)
    assert set(bands["feature"]) == set(NUMERIC_FEATURES)
    for _feature, part in bands.groupby("feature"):
        assert int(part["n"].sum()) == len(features)


def test_band_labels_fall_back_when_bins_are_plain_text(monkeypatch):
    features, target = split_features_target(make_credit_frame(40, seed=16))

    def fake_qcut(series, q, duplicates="drop"):
        return series.astype(int).astype(str)

    monkeypatch.setattr("credit_risk.profile.pd.qcut", fake_qcut)
    bands = numeric_band_default_rates(features, target)
    assert not bands.empty
    assert bands["level"].str.contains("–").sum() == 0


def test_constant_numeric_is_skipped_and_missing_columns_are_ignored():
    features, target = split_features_target(make_credit_frame(40, seed=15))
    features = features.copy()
    features["age"] = 33.0
    bands = numeric_band_default_rates(features, target)
    assert "age" not in set(bands["feature"])

    only_duration = pd.DataFrame({"duration": [6.0, 12.0, 24.0, 36.0]})
    outcome = pd.Series([0, 1, 0, 1])
    assert categorical_default_rates(only_duration, outcome).empty
    assert set(numeric_band_default_rates(only_duration, outcome)["feature"]) == {"duration"}


def test_leading_segment_features():
    importance = pd.DataFrame(
        {"feature": ["duration", "checking_status", "age"], "importance_mean": [0.2, 0.1, 0.0]}
    )
    assert leading_segment_features(importance) == ("checking_status", "duration")
    unknown = pd.DataFrame({"feature": ["not_a_column"], "importance_mean": [0.0]})
    assert leading_segment_features(unknown) == (CATEGORICAL_FEATURES[0], NUMERIC_FEATURES[0])
