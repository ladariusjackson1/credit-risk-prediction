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


def make_classifier(name: str, logistic_c: float | None = None):
    """Return an unfitted classifier. Hyperparameters are read from ``config``.

    ``logistic_c`` overrides the precommitted ``C=1`` on logistic regression.
    The model-class race leaves it unset. The tuned baseline passes a value
    from the fixed grid.
    """
    key = name.strip().lower()
    if logistic_c is not None and key != "logistic_regression":
        raise ValueError("logistic_c applies only to logistic regression.")
    if key == "logistic_regression":
        penalty = 1.0 if logistic_c is None else float(logistic_c)
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("logistic_c must be a positive finite number.")
        return LogisticRegression(
            C=penalty,
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
    logistic_c: float | None = None,
) -> Pipeline:
    """Preprocessing plus classifier. A new pipeline is built on every call."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("model", make_classifier(name, logistic_c=logistic_c)),
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
