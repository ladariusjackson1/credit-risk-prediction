"""Local contributions versus a typical applicant, in plain English."""

from __future__ import annotations

import socket

import pytest

from credit_risk.config import MODEL_PATH, PROJECT_ROOT
from credit_risk.explain import (
    MATERIAL_PROBABILITY,
    ApplicantExplanation,
    FeatureContribution,
    compose_narrative,
    contribution_sentence,
    explain_applicant,
    feature_label,
    format_feature_value,
)
from credit_risk.scoring import ApplicantScore, load_artifact, score_applicant


def _contribution(feature: str, delta: float, label: str | None = None) -> FeatureContribution:
    return FeatureContribution(
        feature=feature,
        label=label or feature_label(feature),
        applicant_value="entered",
        typical_value="typical",
        delta_probability=delta,
    )


def _record(artifact: dict, **overrides):
    record = dict(artifact["reference_applicant"])
    record.update(overrides)
    return record


def test_direction_uses_a_half_point_band():
    assert _contribution("age", MATERIAL_PROBABILITY).direction == "up"
    assert _contribution("age", -MATERIAL_PROBABILITY).direction == "down"
    assert _contribution("age", MATERIAL_PROBABILITY - 1e-6).direction == "flat"
    assert _contribution("age", 0.0).direction == "flat"


def test_value_formatting_and_labels():
    assert format_feature_value("credit_amount", 3972) == "3,972 DM"
    assert format_feature_value("duration", 18) == "18 months"
    assert format_feature_value("age", 33.5) == "33.5 years"
    assert format_feature_value("existing_credits", 2) == "2"
    assert format_feature_value("checking_status", "<0") == "<0"
    assert feature_label("checking_status") == "Checking account status"
    assert feature_label("not_a_field") == "Not a field"


def test_sentences_say_which_way_the_field_pushes():
    up = FeatureContribution(
        feature="checking_status",
        label="Checking account status",
        applicant_value="<0",
        typical_value="no checking",
        delta_probability=0.184,
    )
    down = FeatureContribution(
        feature="savings_status",
        label="Savings",
        applicant_value=">=1000",
        typical_value="<100",
        delta_probability=-0.12,
    )
    assert "raises" in contribution_sentence(up)
    assert "`<0`" in contribution_sentence(up)
    assert "lowers" in contribution_sentence(down)
    assert "12.0 percentage points" in contribution_sentence(down)


def test_narrative_covers_quiet_files_interactions_and_long_lists():
    quiet_lead, quiet_lines, quiet_note = compose_narrative(
        0.208, 0.207, [_contribution("age", 0.0)]
    )
    assert "No field pushes" in quiet_lead
    assert quiet_lines == ()
    assert quiet_note == ""

    gap_lead, gap_lines, _gap_note = compose_narrative(0.30, 0.10, [_contribution("age", 0.0)])
    assert "difference of" in gap_lead
    assert "fields combine" in gap_lines[0]

    small_gap_lead, small_gap_lines, _note = compose_narrative(
        0.201, 0.200, [_contribution("duration", 0.02)]
    )
    assert "difference of" in small_gap_lead
    assert "raises" in small_gap_lines[0]

    many = [_contribution(f"feature_{index}", 0.05 - index * 0.001) for index in range(7)]
    _lead, seven_lines, _note = compose_narrative(0.4, 0.2, many)
    assert any("1 other field" in line for line in seven_lines)
    _lead, eight_lines, _note = compose_narrative(
        0.4, 0.2, many + [_contribution("feature_extra", 0.02)]
    )
    assert any("2 other fields" in line for line in eight_lines)

    personal = FeatureContribution(
        "personal_status", "Personal status", "male single", "female div/dep/mar", 0.03
    )
    foreign = FeatureContribution("foreign_worker", "Foreign worker", "no", "yes", -0.02)
    _lead, lines, one_note = compose_narrative(0.3, 0.2, [personal])
    assert "Personal status is in the saved benchmark model" in one_note
    assert "It should not" in one_note
    noted = ApplicantExplanation(
        score=ApplicantScore(0.30, 0.15, "deny", False, "gradient_boosting"),
        typical_probability=0.20,
        contributions=(personal,),
        lead="This file scores 30.0%.",
        driver_sentences=lines,
        protected_note=one_note,
    )
    assert one_note in noted.narrative
    assert lines[0] in noted.narrative
    _lead, _lines, both_note = compose_narrative(0.3, 0.2, [personal, foreign])
    assert "Personal status and Foreign worker are" in both_note
    assert "They should not" in both_note


def test_reference_profile_has_no_drivers(logistic_artifact):
    explanation = explain_applicant(logistic_artifact["reference_applicant"], logistic_artifact)
    scored = score_applicant(logistic_artifact["reference_applicant"], logistic_artifact)
    assert explanation.score.probability_default == pytest.approx(scored.probability_default)
    assert explanation.score.decision == scored.decision
    assert explanation.typical_probability == pytest.approx(explanation.score.probability_default)
    assert explanation.drivers == ()
    assert explanation.protected_note == ""
    assert "No field pushes" in explanation.narrative
    assert abs(explanation.delta_vs_typical) < 1e-9


def test_one_changed_field_matches_the_counterfactual(logistic_artifact):
    reference = logistic_artifact["reference_applicant"]
    record = _record(logistic_artifact, duration=float(reference["duration"]) + 24)
    explanation = explain_applicant(record, logistic_artifact)
    scored = score_applicant(record, logistic_artifact)
    reset = score_applicant(_record(logistic_artifact), logistic_artifact)
    duration = next(item for item in explanation.contributions if item.feature == "duration")
    age = next(item for item in explanation.contributions if item.feature == "age")

    assert explanation.score.probability_default == pytest.approx(scored.probability_default)
    assert explanation.score.near_cutoff == scored.near_cutoff
    assert duration.delta_probability == pytest.approx(
        scored.probability_default - reset.probability_default
    )
    assert age.delta_probability == pytest.approx(0.0)
    assert age.direction == "flat"
    magnitudes = [abs(item.delta_probability) for item in explanation.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    if duration.direction == "up":
        assert "raises" in explanation.narrative
    elif duration.direction == "down":
        assert "lowers" in explanation.narrative
    else:
        assert "Duration" not in explanation.narrative
    assert "months" in duration.applicant_value


def test_categorical_contribution_is_local_to_the_applicant(logistic_artifact):
    reference = logistic_artifact["reference_applicant"]
    levels = logistic_artifact["category_levels"]["checking_status"]
    other = next(level for level in levels if level != reference["checking_status"])
    purpose_levels = logistic_artifact["category_levels"]["purpose"]
    other_purpose = next(level for level in purpose_levels if level != reference["purpose"])
    record = _record(logistic_artifact, checking_status=other, purpose=other_purpose)
    explanation = explain_applicant(record, logistic_artifact)

    for feature in ("checking_status", "purpose"):
        item = next(row for row in explanation.contributions if row.feature == feature)
        reset_record = dict(record)
        reset_record[feature] = reference[feature]
        expected = (
            explanation.score.probability_default
            - score_applicant(reset_record, logistic_artifact).probability_default
        )
        assert item.delta_probability == pytest.approx(expected)
        assert item.applicant_value == record[feature]
        assert item.typical_value == str(reference[feature])


def test_explain_rejects_a_bad_artifact_or_application(logistic_artifact):
    with pytest.raises(TypeError, match="mapping"):
        explain_applicant(["not", "a", "record"], logistic_artifact)
    with pytest.raises(ValueError, match="Missing"):
        explain_applicant({"age": 30}, logistic_artifact)

    broken = dict(logistic_artifact)
    del broken["reference_applicant"]
    with pytest.raises(KeyError, match="reference_applicant"):
        explain_applicant(_record(logistic_artifact), broken)

    not_a_profile = dict(logistic_artifact)
    not_a_profile["reference_applicant"] = ["age"]
    with pytest.raises(TypeError, match="reference_applicant"):
        explain_applicant(_record(logistic_artifact), not_a_profile)

    incomplete = dict(logistic_artifact)
    incomplete["reference_applicant"] = {"age": 30}
    with pytest.raises(ValueError, match="Missing"):
        explain_applicant(_record(logistic_artifact), incomplete)


def test_committed_model_explains_offline_and_matches_the_study():
    """The desk must score the committed bundle with the network closed."""

    def _offline(*_args, **_kwargs):
        raise AssertionError("scoring tried to open a socket")

    original = socket.socket
    socket.socket = _offline
    try:
        assert MODEL_PATH.is_file()
        assert (PROJECT_ROOT / "data" / "german_credit.csv").is_file()
        artifact = load_artifact()
        typical = explain_applicant(artifact["reference_applicant"], artifact)
        assert typical.drivers == ()
        assert typical.score.decision == "deny"
        assert typical.score.probability_default == pytest.approx(0.2079, abs=0.001)

        savings = dict(artifact["reference_applicant"])
        savings["savings_status"] = ">=1000"
        safer = explain_applicant(savings, artifact)
        savings_effect = next(
            item for item in safer.contributions if item.feature == "savings_status"
        )
        assert savings_effect.direction == "down"
        assert safer.score.decision == "approve"
        assert "lowers" in safer.narrative
        assert "Savings" in safer.narrative
        assert "typical applicant" in safer.narrative

        risky_record = dict(artifact["reference_applicant"])
        risky_record["checking_status"] = "<0"
        risky = explain_applicant(risky_record, artifact)
        checking = next(item for item in risky.contributions if item.feature == "checking_status")
        assert checking.direction == "up"
        assert "raises" in risky.narrative
        assert "`<0`" in risky.narrative
        assert checking.applicant_value == "<0"
        assert checking.typical_value == "no checking"
    finally:
        socket.socket = original
