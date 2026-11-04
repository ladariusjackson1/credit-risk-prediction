"""Three estimators that share one preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_risk import config
from credit_risk.features import build_preprocessor


def make_classifier(name: str):
    """Return an unfitted classifier. Hyperparameters are read from ``config``."""
    key = name.strip().lower()
    if key == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=config.LOGISTIC_MAX_ITER,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=config.RANDOM_STATE,
            n_jobs=1,
        )
    if key == "gradient_boosting":
        # Histogram gradient boosting is scikit-learn's gradient-boosted tree
        # implementation and accepts class_weight, unlike GradientBoostingClassifier.
        return HistGradientBoostingClassifier(
            max_iter=config.GB_MAX_ITER,
            learning_rate=config.GB_LEARNING_RATE,
            max_depth=config.GB_MAX_DEPTH,
            min_samples_leaf=config.GB_MIN_SAMPLES_LEAF,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )
    known = ", ".join(config.MODEL_NAMES)
    raise ValueError(f"Unknown model '{name}'. Choose one of: {known}.")


def build_pipeline(
    name: str,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> Pipeline:
    """Preprocessing plus classifier. A new pipeline is built on every call."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("model", make_classifier(name)),
        ]
    )


def positive_class_probability(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Probability of class 1 (default), independent of ``classes_`` order."""
    model = pipeline.named_steps["model"]
    classes = [int(label) for label in model.classes_]
    if 1 not in classes:
        raise RuntimeError(
            f"The classifier has no positive class (1 = default). Fitted classes: {classes}."
        )
    column = classes.index(1)
    probabilities = pipeline.predict_proba(features)[:, column]
    return np.asarray(probabilities, dtype=float)
