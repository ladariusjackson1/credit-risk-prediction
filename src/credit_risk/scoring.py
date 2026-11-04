"""Score one applicant with a saved pipeline and the cost-minimizing cutoff."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from credit_risk import config
from credit_risk.models import positive_class_probability


@dataclass(frozen=True)
class ApplicantScore:
    """A single desk decision."""

    probability_default: float
    threshold: float
    decision: str
    near_cutoff: bool
    model_name: str


def load_artifact(path: Path | None = None) -> dict:
    """Load the joblib bundle written by ``python -m credit_risk``."""
    source = Path(path) if path is not None else config.MODEL_PATH
    if not source.exists():
        raise FileNotFoundError(f"No trained model at {source}. Run `python -m credit_risk` first.")
    artifact = joblib.load(source)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"{source} is not a credit-risk model artifact.")
    version = artifact.get("artifact_version")
    if version != config.ARTIFACT_VERSION:
        raise ValueError(
            f"Model artifact version is {version}, expected {config.ARTIFACT_VERSION}. "
            "Retrain with `python -m credit_risk`."
        )
    return artifact


def _require(artifact: Mapping, key: str):
    if key not in artifact:
        raise KeyError(f"Model artifact is missing '{key}'. Retrain with `python -m credit_risk`.")
    return artifact[key]


def applicant_frame(record: Mapping, artifact: Mapping) -> pd.DataFrame:
    """Validate one application and return a one-row model frame."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping of feature names to values.")
    columns = list(_require(artifact, "feature_columns"))
    numeric = set(_require(artifact, "numeric_columns"))
    levels = dict(_require(artifact, "category_levels"))
    missing = []
    for column in columns:
        value = record.get(column, None)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing.append(column)
    if missing:
        raise ValueError(f"Missing applicant fields: {missing}.")

    row: dict[str, float | str] = {}
    for column in columns:
        value = record[column]
        if column in numeric:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Field '{column}' must be numeric, got {value!r}.") from exc
            if not math.isfinite(number):
                raise ValueError(f"Field '{column}' must be a finite number, got {value!r}.")
            row[column] = number
        else:
            text = str(value).strip()
            allowed = [str(level) for level in levels.get(column, [])]
            if allowed and text not in allowed:
                raise ValueError(f"Field '{column}' must be one of {allowed}, got {text!r}.")
            row[column] = text
    return pd.DataFrame([row], columns=columns)


def score_applicant(record: Mapping, artifact: Mapping) -> ApplicantScore:
    """Estimate default probability and approve or deny at the saved cutoff."""
    features = applicant_frame(record, artifact)
    probability = float(positive_class_probability(artifact["pipeline"], features)[0])
    threshold = float(_require(artifact, "threshold"))
    band = float(artifact.get("review_band", config.REVIEW_BAND))
    decision = "deny" if probability >= threshold else "approve"
    return ApplicantScore(
        probability_default=probability,
        threshold=threshold,
        decision=decision,
        near_cutoff=abs(probability - threshold) <= band,
        model_name=str(_require(artifact, "model_name")),
    )


def decision_narrative(score: ApplicantScore, artifact: Mapping) -> str:
    """Plain-language reason for the approve or deny decision."""
    action = "declined" if score.decision == "deny" else "approved"
    false_approval = float(_require(artifact, "cost_false_approval"))
    false_denial = float(_require(artifact, "cost_false_denial"))
    review = ""
    if score.near_cutoff:
        review = (
            " The score sits close to the cutoff, so a credit officer should "
            "review the file before it is booked or closed."
        )
    return (
        f"Estimated probability of default is {score.probability_default:.1%}. "
        f"The desk cutoff is {score.threshold:.1%}, chosen to minimize expected cost "
        f"when funding a defaulter costs {false_approval:g} relative to {false_denial:g} "
        f"for turning away a borrower who would have paid. "
        f"This application would be {action}.{review}"
    )
