"""Score a table of applications and attach a plain-English reason to each row."""

from __future__ import annotations

import io
from collections.abc import Mapping

import numpy as np
import pandas as pd

from credit_risk import config
from credit_risk.explain import explain_frame
from credit_risk.scoring import applicant_frame


def batch_template(artifact: Mapping) -> pd.DataFrame:
    """One valid row, using the typical training applicant stored on the model."""
    reference = artifact["reference_applicant"]
    columns = list(artifact["feature_columns"])
    return pd.DataFrame([{column: reference[column] for column in columns}], columns=columns)


def score_batch(
    frame: pd.DataFrame,
    artifact: Mapping,
    max_rows: int = config.MAX_BATCH_ROWS,
) -> pd.DataFrame:
    """Return ``frame`` plus a decision, a probability, and an explanation.

    Extra columns are kept, so a file that still has the ``class`` label can
    be joined back to the score. Rows that fail validation stay in the result
    with a blank score and an ``error`` message. A missing required column,
    a duplicate column, an empty file, or a file over ``max_rows`` rejects
    the whole upload.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a DataFrame.")
    if int(max_rows) < 1:
        raise ValueError("max_rows must be at least 1.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
        raise ValueError(f"CSV has duplicate columns: {duplicates}.")
    if frame.empty:
        raise ValueError("CSV has no rows.")
    if len(frame) > int(max_rows):
        raise ValueError(
            f"CSV has {len(frame)} rows. This desk scores at most {int(max_rows)} at a time."
        )
    columns = list(artifact["feature_columns"])
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}.")

    working = frame.reset_index(drop=True).copy()
    errors: list[str] = []
    valid_positions: list[int] = []
    records: list[dict[str, object]] = []
    for position, row in working.iterrows():
        record = {column: row[column] for column in columns}
        try:
            applicant_frame(record, artifact)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        errors.append("")
        valid_positions.append(int(position))
        records.append(record)

    result = working.copy()
    result["probability_default"] = np.nan
    result["decision"] = ""
    result["near_cutoff"] = pd.Series([pd.NA] * len(working), dtype="boolean")
    result["threshold"] = float(artifact["threshold"])
    result["delta_vs_typical"] = np.nan
    result["explanation"] = ""
    result["error"] = errors

    if records:
        features = pd.DataFrame.from_records(records, columns=columns)
        for position, explanation in zip(
            valid_positions, explain_frame(features, artifact), strict=True
        ):
            result.at[position, "probability_default"] = explanation.score.probability_default
            result.at[position, "decision"] = explanation.score.decision
            result.at[position, "near_cutoff"] = bool(explanation.score.near_cutoff)
            result.at[position, "delta_vs_typical"] = explanation.delta_vs_typical
            result.at[position, "explanation"] = explanation.narrative
    return result


def score_csv(text: str, artifact: Mapping, max_rows: int = config.MAX_BATCH_ROWS) -> str:
    """Parse CSV text, score it, and return CSV text."""
    frame = pd.read_csv(io.StringIO(text))
    return score_batch(frame, artifact, max_rows=max_rows).to_csv(index=False)
