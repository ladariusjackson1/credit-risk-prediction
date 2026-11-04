"""Load the Statlog German Credit file from a local snapshot or from OpenML."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk import config


class DataUnavailableError(FileNotFoundError):
    """The local snapshot is missing and OpenML could not be reached."""


def _to_string_preserve_na(series: pd.Series) -> pd.Series:
    """Strip labels without turning real missing values into the text ``nan``."""
    missing = series.isna()
    text = series.astype(str).str.strip()
    return text.mask(missing, np.nan)


def _to_numeric(series: pd.Series, column: str) -> pd.Series:
    missing = series.isna()
    converted = pd.to_numeric(series, errors="coerce")
    failed = converted.isna() & ~missing
    if bool(failed.any()):
        examples = series.loc[failed].astype(str).unique()[:3].tolist()
        raise ValueError(f"Column '{column}' has non-numeric values, for example {examples}.")
    return converted.astype(float)


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the study columns, clean types, and require good/bad labels.

    Extra columns are ignored. Numeric gaps are left as missing so the model
    imputer can fill them; a value that is present but not numeric is an error.
    """
    required = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}.")

    prepared = pd.DataFrame(index=frame.index)
    for column in config.NUMERIC_FEATURES:
        prepared[column] = _to_numeric(frame[column], column)
    for column in config.CATEGORICAL_FEATURES:
        prepared[column] = _to_string_preserve_na(frame[column])

    target = _to_string_preserve_na(frame[config.TARGET_COLUMN])
    if target.isna().any():
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' contains missing values.")
    target = target.str.lower()
    unknown = sorted(set(target.unique()) - {config.POSITIVE_LABEL, config.NEGATIVE_LABEL})
    if unknown:
        raise ValueError(
            f"Target column '{config.TARGET_COLUMN}' must contain only "
            f"{config.POSITIVE_LABEL!r} and {config.NEGATIVE_LABEL!r}. Found also: {unknown}."
        )
    prepared[config.TARGET_COLUMN] = target
    return prepared[config.STUDY_COLUMNS].reset_index(drop=True)


def encode_target(series: pd.Series) -> pd.Series:
    """Map the dataset label to 1 when the applicant defaulted (``bad``)."""
    cleaned = _to_string_preserve_na(series)
    if cleaned.isna().any():
        raise ValueError("Target contains missing labels.")
    cleaned = cleaned.str.lower()
    unknown = sorted(set(cleaned.unique()) - {config.POSITIVE_LABEL, config.NEGATIVE_LABEL})
    if unknown:
        raise ValueError(f"Unexpected class labels: {unknown}.")
    encoded = (cleaned == config.POSITIVE_LABEL).astype(int)
    encoded.name = "default"
    return encoded


def _download_openml() -> pd.DataFrame:
    """Download OpenML ``credit-g`` version 1. Raises if the network call fails."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(
        config.OPENML_NAME,
        version=config.OPENML_VERSION,
        as_frame=True,
        parser="auto",
    )
    frame = getattr(bunch, "frame", None)
    if frame is None:
        frame = bunch.data.copy()
        frame[config.TARGET_COLUMN] = bunch.target
    else:
        frame = frame.copy()
    frame.columns = [str(column) for column in frame.columns]
    if config.TARGET_COLUMN not in frame.columns and getattr(bunch, "target", None) is not None:
        frame[config.TARGET_COLUMN] = bunch.target.to_numpy()
    return frame


def load_german_credit(cache_path: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Return the study table, preferring the on-disk snapshot.

    The snapshot is the offline fallback. Pass ``refresh=True`` to redownload
    ``credit-g`` from OpenML and replace the snapshot. If a refresh fails and a
    snapshot is already present, that snapshot is used.
    """
    path = Path(cache_path) if cache_path is not None else config.CACHE_PATH
    if path.exists() and not refresh:
        return prepare_frame(pd.read_csv(path))

    try:
        downloaded = _download_openml()
    except Exception as exc:
        if path.exists():
            return prepare_frame(pd.read_csv(path))
        raise DataUnavailableError(
            "Could not download OpenML credit-g and no local snapshot was found at "
            f"{path}. Save german_credit.csv there (see the README) or retry when "
            "OpenML is reachable."
        ) from exc

    prepared = prepare_frame(downloaded)
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(path, index=False)
    return prepared
