"""Batch scoring and CSV explanations."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_risk.batch import batch_template, score_batch, score_csv
from credit_risk.explain import explain_applicant


def _frame(artifact, rows):
    return pd.DataFrame(rows, columns=list(artifact["feature_columns"]))


def test_batch_matches_the_single_applicant_explanation(logistic_artifact):
    reference = dict(logistic_artifact["reference_applicant"])
    changed = dict(reference)
    changed["checking_status"] = "<0"
    changed["duration"] = float(reference["duration"]) + 6
    frame = _frame(logistic_artifact, [reference, changed])
    scored = score_batch(frame, logistic_artifact)
    for position, record in enumerate((reference, changed)):
        explained = explain_applicant(record, logistic_artifact)
        assert scored.loc[position, "probability_default"] == pytest.approx(
            explained.score.probability_default
        )
        assert scored.loc[position, "decision"] == explained.score.decision
        assert scored.loc[position, "explanation"] == explained.narrative
        assert scored.loc[position, "error"] == ""
        assert scored.loc[position, "delta_vs_typical"] == pytest.approx(explained.delta_vs_typical)


def test_invalid_rows_stay_in_the_file(logistic_artifact):
    good = dict(logistic_artifact["reference_applicant"])
    bad = dict(good)
    bad["purpose"] = "yacht"
    frame = _frame(logistic_artifact, [good, bad])
    frame["analyst_note"] = ["keep", "me"]
    scored = score_batch(frame, logistic_artifact)
    assert scored.loc[0, "error"] == ""
    assert "purpose" in scored.loc[1, "error"]
    assert pd.isna(scored.loc[1, "probability_default"])
    assert scored.loc[1, "decision"] == ""
    assert list(scored["analyst_note"]) == ["keep", "me"]


def test_batch_rejects_a_bad_file(logistic_artifact):
    with pytest.raises(TypeError, match="DataFrame"):
        score_batch({"age": 30}, logistic_artifact)
    with pytest.raises(ValueError, match="no rows"):
        score_batch(
            pd.DataFrame(columns=list(logistic_artifact["feature_columns"])), logistic_artifact
        )
    with pytest.raises(ValueError, match="missing columns"):
        score_batch(pd.DataFrame({"age": [30]}), logistic_artifact)
    base = _frame(logistic_artifact, [logistic_artifact["reference_applicant"]])
    first = base.columns[0]
    duplicate = pd.concat([base[[first]], base], axis=1)
    with pytest.raises(ValueError, match="duplicate"):
        score_batch(duplicate, logistic_artifact)
    big = _frame(logistic_artifact, [logistic_artifact["reference_applicant"]] * 3)
    with pytest.raises(ValueError, match="at most 2"):
        score_batch(big, logistic_artifact, max_rows=2)
    with pytest.raises(ValueError, match="at least 1"):
        score_batch(big, logistic_artifact, max_rows=0)


def test_template_and_csv_round_trip(logistic_artifact):
    template = batch_template(logistic_artifact)
    assert list(template.columns) == list(logistic_artifact["feature_columns"])
    text = score_csv(template.to_csv(index=False), logistic_artifact)
    assert "probability_default" in text
    assert "explanation" in text
    broken = dict(logistic_artifact)
    del broken["reference_applicant"]
    with pytest.raises(KeyError, match="reference_applicant"):
        score_batch(template, broken)


def test_empty_feature_frame_explains_nothing(logistic_artifact):
    from credit_risk.explain import explain_frame

    empty = batch_template(logistic_artifact).iloc[0:0]
    assert explain_frame(empty, logistic_artifact) == []
    broken = dict(logistic_artifact)
    broken["reference_applicant"] = ["age"]
    with pytest.raises(TypeError, match="reference_applicant"):
        explain_frame(batch_template(logistic_artifact), broken)
