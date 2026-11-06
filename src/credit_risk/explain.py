"""Plain-English reasons a single applicant scores above or below a typical one.

A local contribution is the change in estimated default probability when that
one field is put back to the typical training applicant and every other field
stays as entered. Positive means the field pushes the score up. The shifts are
not an additive attribution: trees can use combinations of fields, so the
contributions do not have to sum to the gap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_risk import config
from credit_risk.models import positive_class_probability
from credit_risk.scoring import ApplicantScore, applicant_frame, assemble_score

# Half a percentage point. Smaller shifts are left out of the sentences.
MATERIAL_PROBABILITY = 0.005
MAX_DRIVER_SENTENCES = 6

FEATURE_LABELS = {
    "duration": "Duration",
    "credit_amount": "Credit amount",
    "installment_commitment": "Installment rate",
    "residence_since": "Time at residence",
    "age": "Age",
    "existing_credits": "Existing credits",
    "num_dependents": "Dependents",
    "checking_status": "Checking account status",
    "credit_history": "Credit history",
    "purpose": "Purpose",
    "savings_status": "Savings",
    "employment": "Employment tenure",
    "personal_status": "Personal status",
    "other_parties": "Other parties",
    "property_magnitude": "Property",
    "other_payment_plans": "Other installment plans",
    "housing": "Housing",
    "job": "Job",
    "own_telephone": "Telephone",
    "foreign_worker": "Foreign worker",
}

_VALUE_UNITS = {
    "duration": "months",
    "age": "years",
}


@dataclass(frozen=True)
class FeatureContribution:
    """How one field moves the score relative to a typical applicant."""

    feature: str
    label: str
    applicant_value: str
    typical_value: str
    delta_probability: float

    @property
    def direction(self) -> str:
        """``up`` raises default risk, ``down`` lowers it, ``flat`` is immaterial."""
        if self.delta_probability >= MATERIAL_PROBABILITY:
            return "up"
        if self.delta_probability <= -MATERIAL_PROBABILITY:
            return "down"
        return "flat"


@dataclass(frozen=True)
class ApplicantExplanation:
    """Score, local contributions, and the sentences the desk shows."""

    score: ApplicantScore
    typical_probability: float
    contributions: tuple[FeatureContribution, ...]
    lead: str
    driver_sentences: tuple[str, ...]
    protected_note: str

    @property
    def delta_vs_typical(self) -> float:
        return self.score.probability_default - self.typical_probability

    @property
    def drivers(self) -> tuple[FeatureContribution, ...]:
        return tuple(item for item in self.contributions if item.direction != "flat")

    @property
    def narrative(self) -> str:
        parts = [self.lead, *self.driver_sentences]
        if self.protected_note:
            parts.append(self.protected_note)
        return " ".join(part for part in parts if part)


def feature_label(feature: str) -> str:
    """Short name used in sentences."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


def format_feature_value(feature: str, value: object) -> str:
    """Render a model value the way a credit officer would read it."""
    if feature not in config.NUMERIC_FEATURES:
        return str(value)
    number = float(value)
    if feature == "credit_amount":
        return f"{number:,.0f} DM"
    if math.isclose(number, round(number), abs_tol=1e-9):
        rendered = f"{number:.0f}"
    else:
        rendered = f"{number:.1f}"
    unit = _VALUE_UNITS.get(feature)
    if unit:
        return f"{rendered} {unit}"
    return rendered


def contribution_sentence(item: FeatureContribution) -> str:
    """One field, in plain English, against the typical applicant."""
    points = abs(item.delta_probability) * 100
    verb = "raises" if item.direction == "up" else "lowers"
    return (
        f"{item.label} is `{item.applicant_value}` rather than `{item.typical_value}`, "
        f"which {verb} the estimated default probability by {points:.1f} percentage points "
        "versus a typical applicant."
    )


def _overflow_sentence(hidden: int) -> str:
    if hidden == 1:
        return "1 other field also moves the estimated default probability, by a smaller amount."
    return (
        f"{hidden} other fields also move the estimated default probability, by a smaller amount."
    )


def _protected_note(drivers: Sequence[FeatureContribution]) -> str:
    flagged = [item.label for item in drivers if item.feature in config.PROTECTED_FEATURES]
    if not flagged:
        return ""
    if len(flagged) == 1:
        subject, verb, pronoun = flagged[0], "is", "It"
    else:
        subject, verb, pronoun = " and ".join(flagged), "are", "They"
    return (
        f"{subject} {verb} in the saved benchmark model because the 1994 file contains "
        f"{'that column' if len(flagged) == 1 else 'those columns'}. "
        f"{pronoun} should not drive a live U.S. decision."
    )


def compose_narrative(
    probability: float,
    typical_probability: float,
    contributions: Sequence[FeatureContribution],
) -> tuple[str, tuple[str, ...], str]:
    """Lead sentence, driver sentences, and the protected-attribute note."""
    drivers = tuple(item for item in contributions if item.direction != "flat")
    gap = probability - typical_probability
    if abs(gap) < MATERIAL_PROBABILITY and not drivers:
        lead = (
            f"This file scores {probability:.1%}, the same as a typical training applicant "
            f"({typical_probability:.1%}). No field pushes the estimated default probability "
            "up or down relative to that profile."
        )
        return lead, (), ""

    lead = (
        f"This file scores {probability:.1%}. A typical training applicant scores "
        f"{typical_probability:.1%}, a difference of {gap:+.1%}. "
        "Each line is a local contribution versus that typical applicant: "
        "how much the estimated default probability changes when that one field "
        "is put back, and the other fields stay as entered."
    )
    shown = drivers[:MAX_DRIVER_SENTENCES]
    sentences = [contribution_sentence(item) for item in shown]
    hidden = len(drivers) - len(shown)
    if hidden:
        sentences.append(_overflow_sentence(hidden))
    elif not drivers:
        sentences.append(
            "No single field moves the estimated default probability on its own. "
            "The gap comes from how the fields combine."
        )
    return lead, tuple(sentences), _protected_note(drivers)


def _values_match(column: str, left: object, right: object, numeric: set[str]) -> bool:
    if column in numeric:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    return str(left) == str(right)


def explain_frame(features: pd.DataFrame, artifact: Mapping) -> list[ApplicantExplanation]:
    """Score each row and describe it against the typical training applicant.

    Contributions match :func:`explain_applicant`. Scoring the rows together,
    then one counterfactual column at a time, keeps a CSV of a few thousand
    rows within a short request.
    """
    if "reference_applicant" not in artifact:
        raise KeyError(
            "Model artifact is missing 'reference_applicant'. Retrain with `python -m credit_risk`."
        )
    reference = artifact["reference_applicant"]
    if not isinstance(reference, Mapping):
        raise TypeError("reference_applicant must be a mapping of feature names to values.")
    if features.empty:
        return []

    columns = list(artifact["feature_columns"])
    frame = features.loc[:, columns].reset_index(drop=True)
    typical = applicant_frame(reference, artifact)
    typical_row = typical.iloc[0]
    numeric = set(artifact["numeric_columns"])
    pipeline = artifact["pipeline"]
    probabilities = positive_class_probability(pipeline, frame)
    typical_probability = float(positive_class_probability(pipeline, typical)[0])
    deltas = np.zeros((len(frame), len(columns)), dtype=float)

    for index, column in enumerate(columns):
        reference_value = typical_row[column]
        counter = frame.copy()
        if column in numeric:
            reference_number = float(reference_value)
            same = np.isclose(
                frame[column].to_numpy(dtype=float),
                reference_number,
                rtol=0.0,
                atol=1e-9,
            )
            counter[column] = reference_number
        else:
            reference_text = str(reference_value)
            same = frame[column].astype(str).to_numpy() == reference_text
            counter[column] = reference_text
        if bool(np.all(same)):
            continue
        counter_probability = positive_class_probability(pipeline, counter)
        delta = probabilities - counter_probability
        delta[same] = 0.0
        deltas[:, index] = delta

    explanations: list[ApplicantExplanation] = []
    for row_index, row in frame.iterrows():
        contributions: list[FeatureContribution] = []
        for column_index, column in enumerate(columns):
            contributions.append(
                FeatureContribution(
                    feature=column,
                    label=feature_label(column),
                    applicant_value=format_feature_value(column, row[column]),
                    typical_value=format_feature_value(column, typical_row[column]),
                    delta_probability=float(deltas[int(row_index), column_index]),
                )
            )
        ordered = tuple(
            sorted(contributions, key=lambda item: (-abs(item.delta_probability), item.feature))
        )
        probability = float(probabilities[int(row_index)])
        lead, sentences, protected_note = compose_narrative(
            probability, typical_probability, ordered
        )
        explanations.append(
            ApplicantExplanation(
                score=assemble_score(probability, artifact),
                typical_probability=typical_probability,
                contributions=ordered,
                lead=lead,
                driver_sentences=sentences,
                protected_note=protected_note,
            )
        )
    return explanations


def explain_applicant(record: Mapping, artifact: Mapping) -> ApplicantExplanation:
    """Score one applicant and describe each field against the typical profile.

    The typical profile is the training median for numeric fields and the most
    common label for categories, stored on the model artifact.
    """
    if "reference_applicant" not in artifact:
        raise KeyError(
            "Model artifact is missing 'reference_applicant'. Retrain with `python -m credit_risk`."
        )
    reference = artifact["reference_applicant"]
    if not isinstance(reference, Mapping):
        raise TypeError("reference_applicant must be a mapping of feature names to values.")

    applicant = applicant_frame(record, artifact)
    typical = applicant_frame(reference, artifact)
    columns = list(applicant.columns)
    numeric = set(artifact["numeric_columns"])
    applicant_row = applicant.iloc[0]
    typical_row = typical.iloc[0]

    rows: list[dict[str, object]] = [applicant_row.to_dict(), typical_row.to_dict()]
    changed: list[str] = []
    for column in columns:
        if _values_match(column, applicant_row[column], typical_row[column], numeric):
            continue
        counter = applicant_row.to_dict()
        counter[column] = typical_row[column]
        rows.append(counter)
        changed.append(column)

    frame = pd.DataFrame.from_records(rows, columns=columns)
    probabilities = positive_class_probability(artifact["pipeline"], frame)
    probability = float(probabilities[0])
    typical_probability = float(probabilities[1])
    counterfactual = {
        feature: float(probabilities[index + 2]) for index, feature in enumerate(changed)
    }

    contributions: list[FeatureContribution] = []
    for column in columns:
        delta = 0.0
        if column in counterfactual:
            delta = probability - counterfactual[column]
        contributions.append(
            FeatureContribution(
                feature=column,
                label=feature_label(column),
                applicant_value=format_feature_value(column, applicant_row[column]),
                typical_value=format_feature_value(column, typical_row[column]),
                delta_probability=delta,
            )
        )
    ordered = tuple(
        sorted(contributions, key=lambda item: (-abs(item.delta_probability), item.feature))
    )
    lead, sentences, protected_note = compose_narrative(probability, typical_probability, ordered)
    return ApplicantExplanation(
        score=assemble_score(probability, artifact),
        typical_probability=typical_probability,
        contributions=ordered,
        lead=lead,
        driver_sentences=sentences,
        protected_note=protected_note,
    )
