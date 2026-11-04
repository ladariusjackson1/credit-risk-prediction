"""Loader, snapshot fallback, and target encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import CACHE_PATH
from credit_risk.data import (
    DataUnavailableError,
    _download_openml,
    encode_target,
    load_german_credit,
    prepare_frame,
)
from tests.synthetic import make_credit_frame


def test_prepare_and_encode_round_trip():
    frame = make_credit_frame(30, seed=1)
    frame.loc[0, "class"] = " Bad "
    frame.loc[1, "purpose"] = " radio/tv "
    prepared = prepare_frame(frame)
    assert prepared.loc[0, "class"] == "bad"
    assert prepared.loc[1, "purpose"] == "radio/tv"
    encoded = encode_target(prepared["class"])
    assert encoded.tolist() == (prepared["class"] == "bad").astype(int).tolist()
    assert encoded.name == "default"


def test_prepare_rejects_bad_inputs():
    frame = make_credit_frame(12, seed=2)
    missing = frame.drop(columns=["age"])
    with pytest.raises(ValueError, match="missing columns"):
        prepare_frame(missing)

    bad_number = frame.copy()
    bad_number["age"] = bad_number["age"].astype(object)
    bad_number.loc[0, "age"] = "forty"
    with pytest.raises(ValueError, match="non-numeric"):
        prepare_frame(bad_number)

    bad_label = frame.copy()
    bad_label.loc[0, "class"] = "ugly"
    with pytest.raises(ValueError, match="ugly"):
        prepare_frame(bad_label)

    blank_target = frame.copy()
    blank_target.loc[0, "class"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        prepare_frame(blank_target)

    with pytest.raises(ValueError, match="Unexpected"):
        encode_target(pd.Series(["good", "maybe"]))
    with pytest.raises(ValueError, match="missing"):
        encode_target(pd.Series(["good", None]))


def test_cache_is_preferred_and_refresh_can_fail_open(tmp_path, monkeypatch):
    frame = make_credit_frame(18, seed=4)
    path = tmp_path / "german_credit.csv"
    prepare_frame(frame).to_csv(path, index=False)

    def explode():
        raise AssertionError("download should not run")

    monkeypatch.setattr("credit_risk.data._download_openml", explode)
    loaded = load_german_credit(cache_path=path, refresh=False)
    assert len(loaded) == 18
    assert set(loaded["class"]) <= {"good", "bad"}

    def boom():
        raise RuntimeError("openml down")

    monkeypatch.setattr("credit_risk.data._download_openml", boom)
    fallback = load_german_credit(cache_path=path, refresh=True)
    assert len(fallback) == 18

    with pytest.raises(DataUnavailableError, match="german_credit.csv"):
        load_german_credit(cache_path=tmp_path / "missing.csv", refresh=True)


def test_refresh_writes_snapshot(tmp_path, monkeypatch):
    frame = make_credit_frame(16, seed=5)

    monkeypatch.setattr("credit_risk.data._download_openml", lambda: frame)
    path = tmp_path / "nested" / "german_credit.csv"
    loaded = load_german_credit(cache_path=path, refresh=True)
    assert path.exists()
    assert len(loaded) == 16
    again = pd.read_csv(path)
    assert list(again.columns)[:3] == ["checking_status", "duration", "credit_history"]


def test_openml_shapes(monkeypatch):
    raw = make_credit_frame(14, seed=6)

    class WithFrame:
        frame = raw
        target = raw["class"]

    class SplitParts:
        frame = None
        data = raw.drop(columns=["class"])
        target = raw["class"]

    class FrameMissingTarget:
        def __init__(self):
            self.frame = raw.drop(columns=["class"])
            self.target = raw["class"]

    monkeypatch.setattr("sklearn.datasets.fetch_openml", lambda *args, **kwargs: WithFrame())
    assert "class" in _download_openml().columns

    monkeypatch.setattr("sklearn.datasets.fetch_openml", lambda *args, **kwargs: SplitParts())
    from_parts = _download_openml()
    assert from_parts["class"].astype(str).tolist() == raw["class"].tolist()

    monkeypatch.setattr(
        "sklearn.datasets.fetch_openml", lambda *args, **kwargs: FrameMissingTarget()
    )
    assert "class" in _download_openml().columns


def test_shipped_snapshot_matches_statlog_file():
    if not CACHE_PATH.exists():
        pytest.skip("Snapshot is written by python -m credit_risk")
    frame = load_german_credit(cache_path=CACHE_PATH)
    assert len(frame) == 1000
    assert frame["class"].eq("bad").mean() == pytest.approx(0.3)
    assert frame.loc[0, "duration"] == pytest.approx(6)
    assert frame.loc[0, "credit_amount"] == pytest.approx(1169)
    assert frame.loc[0, "class"] == "good"
    assert frame.loc[0, "checking_status"] == "<0"
