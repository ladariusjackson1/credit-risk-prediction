"""Percentile bootstrap intervals for scores that are already in hand.

The resample draws inside each class so every replicate still has defaults and
paying applicants. The model is not refit. A tuned baseline's interval also
does not repeat the choice of its penalty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from credit_risk.evaluate import expected_cost, predict_at_threshold

BOOTSTRAP_METHOD = (
    "Stratified percentile bootstrap of fixed scores. Each replicate draws "
    "applicants with replacement inside the default class and inside the paid "
    "class, keeping those class counts. The model is not refit. The interval "
    "does not include the noise of retraining or, for the tuned logistic "
    "baseline, the noise of choosing C again."
)

METRIC_NAMES = ("roc_auc", "average_precision", "brier", "cost_per_applicant")


def _ranking_metrics(actual: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(actual, scores)),
        "average_precision": float(average_precision_score(actual, scores)),
        "brier": float(brier_score_loss(actual, scores)),
    }


def _stratified_index(pools: dict[int, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    parts = [rng.choice(pools[label], size=len(pools[label]), replace=True) for label in (0, 1)]
    return np.concatenate(parts)


def auc_difference_sentence(
    model_name: str,
    baseline_name: str,
    estimate: float,
    ci_low: float,
    ci_high: float,
) -> str:
    """Plain sentence for the out-of-fold ROC-AUC gap. Uses only the numbers passed in."""
    interval = f"{ci_low:.3f} to {ci_high:.3f}"
    gap = f"{estimate:+.3f}"
    if ci_low <= 0.0 <= ci_high:
        return (
            f"On out-of-fold scores, {model_name} minus {baseline_name} has a ROC-AUC "
            f"difference of {gap} (95% bootstrap interval {interval}). "
            "The interval includes zero, so this training sample does not separate them "
            "on ROC-AUC."
        )
    direction = "higher" if estimate > 0 else "lower"
    return (
        f"On out-of-fold scores, {model_name} ROC-AUC is {gap} versus {baseline_name} "
        f"(95% bootstrap interval {interval}), which is {direction}. "
        "The interval excludes zero."
    )


def bootstrap_scores(
    y_true,
    scores_by_model: Mapping[str, Sequence[float]],
    n_boot: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
    baseline: str | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Point estimates and percentile intervals for each model's fixed scores.

    When ``baseline`` is set, the table also contains each other model minus
    that baseline, on the same replicates. ``thresholds`` adds cost per
    applicant at a cutoff that stays fixed across replicates.
    """
    if n_boot < 20:
        raise ValueError("n_boot must be at least 20.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    actual = np.asarray(y_true).astype(int)
    if len(np.unique(actual)) < 2:
        raise ValueError("Bootstrap metrics require both classes.")
    models: dict[str, np.ndarray] = {}
    for name, scores in scores_by_model.items():
        array = np.asarray(scores, dtype=float)
        if len(array) != len(actual):
            raise ValueError(f"Scores for '{name}' do not match y_true.")
        models[name] = array
    if not models:
        raise ValueError("scores_by_model is empty.")
    if baseline is not None and baseline not in models:
        raise ValueError(f"Baseline '{baseline}' is not in the scores.")
    cutoff = dict(thresholds or {})
    missing_cutoff = [name for name in models if name not in cutoff]
    if thresholds is not None and missing_cutoff:
        raise ValueError(f"Missing thresholds for {missing_cutoff}.")

    point: dict[str, dict[str, float]] = {}
    for name, scores in models.items():
        metrics = _ranking_metrics(actual, scores)
        if thresholds is not None:
            decision = predict_at_threshold(scores, cutoff[name])
            metrics["cost_per_applicant"] = expected_cost(actual, decision) / len(actual)
        point[name] = metrics

    metric_names = [name for name in METRIC_NAMES if name in next(iter(point.values()))]
    replicates = {
        name: {metric: np.empty(n_boot, dtype=float) for metric in metric_names} for name in models
    }
    rng = np.random.default_rng(random_state)
    pools = {label: np.flatnonzero(actual == label) for label in (0, 1)}
    for draw in range(n_boot):
        index = _stratified_index(pools, rng)
        sample = actual[index]
        for name, scores in models.items():
            metrics = _ranking_metrics(sample, scores[index])
            if thresholds is not None:
                decision = predict_at_threshold(scores[index], cutoff[name])
                metrics["cost_per_applicant"] = expected_cost(sample, decision) / len(sample)
            for metric in metric_names:
                replicates[name][metric][draw] = metrics[metric]

    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - alpha / 2.0)
    rows: list[dict[str, object]] = []

    def _row(comparison: str, name: str, metric: str, estimate: float, sample: np.ndarray) -> dict:
        ci_low = float(np.percentile(sample, low_q))
        ci_high = float(np.percentile(sample, high_q))
        return {
            "comparison": comparison,
            "model": name,
            "baseline": baseline,
            "metric": metric,
            "estimate": float(estimate),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "includes_zero": bool(ci_low <= 0.0 <= ci_high),
            "n_boot": int(n_boot),
            "alpha": float(alpha),
        }

    for name, metrics in point.items():
        for metric in metric_names:
            rows.append(_row("level", name, metric, metrics[metric], replicates[name][metric]))
    if baseline is not None:
        for name in models:
            if name == baseline:
                continue
            for metric in metric_names:
                estimate = point[name][metric] - point[baseline][metric]
                sample = replicates[name][metric] - replicates[baseline][metric]
                rows.append(_row("difference", name, metric, estimate, sample))
    return pd.DataFrame(rows)
