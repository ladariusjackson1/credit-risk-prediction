"""Single-applicant scoring and the saved model bundle."""

from __future__ import annotations

import joblib
import pytest

from credit_risk.scoring import decision_narrative, load_artifact, score_applicant


def _record(artifact: dict, **overrides):
    record = dict(artifact["reference_applicant"])
    record.update(overrides)
    return record


def test_decisions_follow_the_cutoff(logistic_artifact):
    record = _record(logistic_artifact)
    scored = score_applicant(record, logistic_artifact)
    assert 0.0 < scored.probability_default < 1.0
    assert scored.model_name == "logistic_regression"

    deny_all = dict(logistic_artifact)
    deny_all["threshold"] = 0.0
    deny_all["review_band"] = 0.0
    denied = score_applicant(record, deny_all)
    assert denied.decision == "deny"

    approve = dict(logistic_artifact)
    approve["threshold"] = 1.0
    approve["review_band"] = 0.0
    approved = score_applicant(record, approve)
    assert approved.decision == "approve"
    assert approved.near_cutoff is False
    assert "approved" in decision_narrative(approved, approve)
    assert "credit officer" not in decision_narrative(approved, approve)

    on_the_line = dict(logistic_artifact)
    on_the_line["threshold"] = scored.probability_default
    on_the_line["review_band"] = 0.05
    borderline = score_applicant(record, on_the_line)
    assert borderline.decision == "deny"
    assert borderline.near_cutoff is True
    assert "declined" in decision_narrative(borderline, on_the_line)
    assert "credit officer" in decision_narrative(borderline, on_the_line)

    record_with_extra = _record(logistic_artifact, analyst_note="ignore me")
    assert score_applicant(
        record_with_extra, logistic_artifact
    ).probability_default == pytest.approx(scored.probability_default)


def test_scoring_rejects_bad_applications(logistic_artifact):
    with pytest.raises(TypeError, match="mapping"):
        score_applicant(["not", "a", "record"], logistic_artifact)
    with pytest.raises(ValueError, match="Missing"):
        score_applicant({"age": 30}, logistic_artifact)
    with pytest.raises(ValueError, match="numeric"):
        score_applicant(_record(logistic_artifact, duration="long"), logistic_artifact)
    with pytest.raises(ValueError, match="finite"):
        score_applicant(_record(logistic_artifact, age=float("inf")), logistic_artifact)
    with pytest.raises(ValueError, match="Missing"):
        score_applicant(_record(logistic_artifact, age=float("nan")), logistic_artifact)
    with pytest.raises(ValueError, match="must be one of"):
        score_applicant(_record(logistic_artifact, purpose="yacht"), logistic_artifact)

    broken = dict(logistic_artifact)
    del broken["cost_false_approval"]
    scored = score_applicant(_record(logistic_artifact), logistic_artifact)
    with pytest.raises(KeyError, match="cost_false_approval"):
        decision_narrative(scored, broken)


def test_load_artifact_round_trip(tmp_path, logistic_artifact):
    path = tmp_path / "model.joblib"
    joblib.dump(logistic_artifact, path)
    loaded = load_artifact(path)
    assert loaded["model_name"] == "logistic_regression"

    missing = tmp_path / "absent.joblib"
    with pytest.raises(FileNotFoundError, match="python -m credit_risk"):
        load_artifact(missing)

    joblib.dump({"weights": [1, 2, 3]}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError, match="not a credit-risk"):
        load_artifact(tmp_path / "bad.joblib")

    stale = dict(logistic_artifact)
    stale["artifact_version"] = 99
    joblib.dump(stale, tmp_path / "stale.joblib")
    with pytest.raises(ValueError, match="version"):
        load_artifact(tmp_path / "stale.joblib")
