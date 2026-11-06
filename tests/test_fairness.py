"""Group approval and error rates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.fairness import (
    age_band,
    fairness_note,
    fairness_report,
    sex_from_personal_status,
)


def test_labels_follow_the_codebook():
    assert sex_from_personal_status("female div/dep/mar") == "female"
    assert sex_from_personal_status("male single") == "male"
    assert sex_from_personal_status("other") == "unknown"
    assert sex_from_personal_status(None) == "unknown"
    assert sex_from_personal_status(float("nan")) == "unknown"
    assert age_band(24) == "under 25"
    assert age_band(25) == "25 and older"
    assert age_band(float("nan")) == "unknown"
    assert age_band(float("inf")) == "unknown"


def test_gaps_match_a_hand_count():
    labels = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.2, 0.8, 0.4, 0.6])
    status = [
        "female div/dep/mar",
        "female div/dep/mar",
        "female div/dep/mar",
        "male single",
        "male single",
        "male mar/wid",
    ]
    ages = [22, 30, 40, 21, 33, 44]
    report = fairness_report(
        labels,
        scores,
        ages,
        status,
        threshold=0.5,
        split_label="worked example",
    )
    rates = report["rates"]
    female = rates.loc[(rates["family"] == "sex") & (rates["group"] == "female")].iloc[0]
    male = rates.loc[(rates["family"] == "sex") & (rates["group"] == "male")].iloc[0]
    assert female["approval_rate"] == pytest.approx(2 / 3)
    assert female["false_approval_rate"] == pytest.approx(0.5)
    assert female["false_denial_rate"] == pytest.approx(0.0)
    assert male["approval_rate"] == pytest.approx(1 / 3)
    assert male["false_approval_rate"] == pytest.approx(1.0)
    assert male["false_denial_rate"] == pytest.approx(1.0)
    sex_gap = report["gaps"].loc[report["gaps"]["family"] == "sex"].iloc[0]
    assert sex_gap["approval_rate_gap"] == pytest.approx(1 / 3)
    assert sex_gap["approval_ratio_a_over_b"] == pytest.approx(2.0)
    young = rates.loc[(rates["family"] == "age") & (rates["group"] == "under 25")]
    assert int(young.iloc[0]["n"]) == 2
    note = report["note"]
    assert "66.7%" in note
    assert "33.3%" in note
    assert "+33.3 percentage points" in note
    assert "not a fair-lending review" in note
    assert "worked example" in note


def test_missing_group_is_described_without_a_rate():
    report = fairness_report(
        np.array([0, 1, 0, 1]),
        np.array([0.2, 0.8, 0.1, 0.9]),
        [40, 41, 42, 43],
        ["male single", "male mar/wid", "male div/sep", "not a status"],
        threshold=0.5,
        split_label="no young applicants",
    )
    assert "not estimated" in report["note"]
    assert "left out of the sex gap" in report["note"]
    broken = report["gaps"].drop(index=report["gaps"].index[:1])
    with pytest.raises(ValueError, match="sex row"):
        fairness_note(
            report["rates"],
            broken,
            threshold=0.5,
            split_label="no young applicants",
            n_rows=4,
            age_cut=25,
        )


def test_a_group_with_no_defaulters_says_the_rate_is_not_estimated():
    report = fairness_report(
        np.array([0, 0, 1, 0]),
        np.array([0.2, 0.3, 0.8, 0.1]),
        [30, 31, 32, 33],
        ["female div/dep/mar", "female div/dep/mar", "male single", "male single"],
        threshold=0.5,
        split_label="no female defaults",
    )
    assert "not estimated" in report["note"]


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        fairness_report(
            [0, 1], [0.2], [30, 40], ["male single", "female div/dep/mar"], 0.5, split_label="x"
        )


def test_zero_approval_ratio_is_not_estimated():
    report = fairness_report(
        np.array([1, 0, 1, 0]),
        np.array([0.1, 0.2, 0.9, 0.8]),
        [22, 23, 40, 41],
        ["female div/dep/mar", "female div/dep/mar", "male single", "male single"],
        threshold=0.5,
        split_label="women approved and men denied",
    )
    # Women: scores 0.1 and 0.2 are both approved. Men: 0.9 and 0.8 are both denied.
    sex_gap = report["gaps"].loc[report["gaps"]["family"] == "sex"].iloc[0]
    assert sex_gap["approval_rate_b"] == pytest.approx(0.0)
    assert pd.isna(sex_gap["approval_ratio_a_over_b"])
    assert "approval ratio is not estimated" in report["note"]
