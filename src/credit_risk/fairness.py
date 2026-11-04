"""Approval and error rates by age and by the file's sex label.

``personal_status`` mixes sex and marital status. The sex rows collapse that
column on the first word. The age cut is a reporting choice. Neither split is
a fair-lending test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from credit_risk import config
from credit_risk.evaluate import predict_at_threshold


def age_band_labels(cut: float = config.AGE_GROUP_CUT) -> tuple[str, str]:
    """Younger band, then the band at or above the cut."""
    text = f"{float(cut):g}"
    return f"under {text}", f"{text} and older"


def sex_from_personal_status(value: object) -> str:
    """Collapse the Statlog personal-status label to female, male, or unknown."""
    if _missing(value):
        return "unknown"
    text = str(value).strip().lower()
    if text.startswith("female"):
        return "female"
    if text.startswith("male"):
        return "male"
    return "unknown"


def age_band(value: object, cut: float = config.AGE_GROUP_CUT) -> str:
    """Two bands split at ``cut``. Missing ages stay out of both bands."""
    younger, older = age_band_labels(cut)
    if _missing(value):
        return "unknown"
    number = float(value)
    if not math.isfinite(number):
        return "unknown"
    if number < float(cut):
        return younger
    return older


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _group_rates(
    actual: np.ndarray,
    approved: np.ndarray,
    labels: pd.Series,
    family: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = pd.Series(labels).astype(str).reset_index(drop=True)
    grouped = labels.groupby(labels, sort=False).groups
    for level, positions in grouped.items():
        index = np.asarray(list(positions), dtype=int)
        outcome = actual[index]
        booked = approved[index]
        n = int(len(index))
        n_default = int(outcome.sum())
        n_paid = n - n_default
        n_approved = int(booked.sum())
        false_approvals = int((booked & (outcome == 1)).sum())
        false_denials = int((~booked & (outcome == 0)).sum())
        rows.append(
            {
                "family": family,
                "group": str(level),
                "n": n,
                "n_default": n_default,
                "n_paid": n_paid,
                "approvals": n_approved,
                "approval_rate": n_approved / n if n else np.nan,
                "false_approval_rate": false_approvals / n_default if n_default else np.nan,
                "false_denial_rate": false_denials / n_paid if n_paid else np.nan,
                "default_rate": n_default / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _gap_row(rates: pd.DataFrame, family: str, group_a: str, group_b: str) -> dict[str, object]:
    def _one(group: str) -> pd.Series | None:
        match = rates.loc[(rates["family"] == family) & (rates["group"] == group)]
        if match.empty:
            return None
        return match.iloc[0]

    left = _one(group_a)
    right = _one(group_b)
    row: dict[str, object] = {
        "family": family,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(left["n"]) if left is not None else 0,
        "n_b": int(right["n"]) if right is not None else 0,
    }
    for metric in ("approval_rate", "false_approval_rate", "false_denial_rate"):
        left_value = float(left[metric]) if left is not None else np.nan
        right_value = float(right[metric]) if right is not None else np.nan
        row[f"{metric}_a"] = left_value
        row[f"{metric}_b"] = right_value
        if math.isfinite(left_value) and math.isfinite(right_value):
            row[f"{metric}_gap"] = left_value - right_value
        else:
            row[f"{metric}_gap"] = np.nan
    left_approval = float(row["approval_rate_a"])
    right_approval = float(row["approval_rate_b"])
    if math.isfinite(left_approval) and math.isfinite(right_approval) and right_approval > 0:
        row["approval_ratio_a_over_b"] = left_approval / right_approval
    else:
        row["approval_ratio_a_over_b"] = np.nan
    return row


def _percent(value: object) -> str:
    number = float(value) if value is not None else np.nan
    if not math.isfinite(number):
        return "not estimated"
    return f"{number:.1%}"


def _points(value: object) -> str:
    number = float(value) if value is not None else np.nan
    if not math.isfinite(number):
        return "not estimated"
    return f"{number * 100:+.1f} percentage points"


def _pair_paragraph(
    rates: pd.DataFrame,
    gap: dict[str, object],
    left_name: str,
    right_name: str,
) -> str:
    family = str(gap["family"])
    left = rates.loc[(rates["family"] == family) & (rates["group"] == gap["group_a"])]
    right = rates.loc[(rates["family"] == family) & (rates["group"] == gap["group_b"])]
    if left.empty or right.empty:
        return (
            f"The {family} comparison of {left_name} versus {right_name} is not estimated "
            "because one of those groups has no applicants on this split."
        )
    left_row = left.iloc[0]
    right_row = right.iloc[0]
    approval = (
        f"On this split, {left_name} were approved {_percent(left_row['approval_rate'])} "
        f"of the time (n={int(left_row['n'])}) and {right_name} were approved "
        f"{_percent(right_row['approval_rate'])} of the time (n={int(right_row['n'])}). "
        f"The approval-rate gap, {left_name} minus {right_name}, is "
        f"{_points(gap['approval_rate_gap'])}."
    )
    false_approval = (
        f" Among people who defaulted, the share who were still approved was "
        f"{_percent(left_row['false_approval_rate'])} for {left_name} "
        f"({int(left_row['n_default'])} defaulters) and "
        f"{_percent(right_row['false_approval_rate'])} for {right_name} "
        f"({int(right_row['n_default'])} defaulters). "
        f"That gap is {_points(gap['false_approval_rate_gap'])}."
    )
    false_denial = (
        f" Among people who paid, the share who were turned away was "
        f"{_percent(left_row['false_denial_rate'])} for {left_name} "
        f"({int(left_row['n_paid'])} who paid) and "
        f"{_percent(right_row['false_denial_rate'])} for {right_name} "
        f"({int(right_row['n_paid'])} who paid). "
        f"That gap is {_points(gap['false_denial_rate_gap'])}."
    )
    ratio = float(gap["approval_ratio_a_over_b"])
    if math.isfinite(ratio):
        ratio_sentence = (
            f" The approval ratio of {left_name} to {right_name} is {ratio:.2f}. "
            "A ratio of 0.80 is a common U.S. screening rule of thumb, and it is not a "
            "legal result on this file."
        )
    else:
        ratio_sentence = (
            " The approval ratio is not estimated because the comparison group has an "
            "approval rate of zero or the rate itself is missing."
        )
    return approval + false_approval + false_denial + ratio_sentence


def fairness_note(
    rates: pd.DataFrame,
    gaps: pd.DataFrame,
    *,
    threshold: float,
    split_label: str,
    n_rows: int,
    age_cut: float,
) -> str:
    """Describe the gaps in plain English. Every figure comes from ``rates`` and ``gaps``."""
    younger, older = age_band_labels(age_cut)
    sex_gap = gaps.loc[gaps["family"] == "sex"]
    age_gap = gaps.loc[gaps["family"] == "age"]
    if sex_gap.empty or age_gap.empty:
        raise ValueError("Gaps must include a sex row and an age row.")
    intro = (
        f"At cutoff {threshold:.2f} on the {split_label} ({n_rows} applicants), "
        "a file is approved when its score is below the cutoff."
    )
    sex_text = _pair_paragraph(
        rates,
        sex_gap.iloc[0].to_dict(),
        "female-labeled applicants",
        "male-labeled applicants",
    )
    age_text = _pair_paragraph(
        rates,
        age_gap.iloc[0].to_dict(),
        f"applicants {younger}",
        f"applicants {older}",
    )
    unknown = rates.loc[(rates["family"] == "sex") & (rates["group"] == "unknown")]
    unknown_sentence = ""
    if not unknown.empty and int(unknown.iloc[0]["n"]) > 0:
        unknown_sentence = (
            f" {int(unknown.iloc[0]['n'])} applicants had a personal_status value that "
            "does not start with female or male, and they are left out of the sex gap."
        )
    limits = (
        " Read these gaps as a description of this cutoff on this split, not as a finding "
        "of discrimination. personal_status in the 1994 file mixes sex and marital status, "
        f"so the female and male rows collapse those labels. The age split is {younger} versus "
        f"{older}, which is a reporting choice. Age is an input to the model, and personal "
        "status is an input to the benchmark model. A cell with few defaults moves when one "
        "or two people change outcome. This is not a fair-lending review, not a U.S. applicant "
        "population, and not a reason by itself to approve or deny anyone."
    )
    return "\n\n".join(
        [
            intro,
            sex_text + unknown_sentence,
            age_text,
            limits.strip(),
        ]
    )


def fairness_report(
    y_true,
    proba,
    age,
    personal_status,
    threshold: float,
    *,
    split_label: str,
    age_cut: float = config.AGE_GROUP_CUT,
) -> dict[str, object]:
    """Group rates, gaps, and the plain-English note at one cutoff."""
    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    ages = pd.Series(age).reset_index(drop=True)
    status = pd.Series(personal_status).reset_index(drop=True)
    if not (len(actual) == len(scores) == len(ages) == len(status)):
        raise ValueError("y_true, proba, age, and personal_status must have the same length.")
    approved = predict_at_threshold(scores, threshold) == 0
    sex_labels = status.map(sex_from_personal_status)
    age_labels = ages.map(lambda value: age_band(value, age_cut))
    rates = pd.concat(
        [
            _group_rates(actual, approved, sex_labels, "sex"),
            _group_rates(actual, approved, age_labels, "age"),
        ],
        ignore_index=True,
    )
    younger, older = age_band_labels(age_cut)
    gaps = pd.DataFrame(
        [
            _gap_row(rates, "sex", "female", "male"),
            _gap_row(rates, "age", younger, older),
        ]
    )
    note = fairness_note(
        rates,
        gaps,
        threshold=float(threshold),
        split_label=split_label,
        n_rows=len(actual),
        age_cut=float(age_cut),
    )
    return {
        "split": split_label,
        "n": int(len(actual)),
        "threshold": float(threshold),
        "age_cut": float(age_cut),
        "rates": rates,
        "gaps": gaps,
        "note": note,
    }
