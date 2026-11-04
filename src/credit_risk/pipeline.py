"""Train the comparison, choose a cutoff on out-of-fold scores, and write the report."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from credit_risk import config
from credit_risk.data import load_german_credit
from credit_risk.evaluate import (
    baseline_costs,
    choose_threshold,
    cost_ratio_sensitivity,
    permutation_importance_table,
    row_at_threshold,
    score_at_threshold,
    threshold_sweep,
)
from credit_risk.features import (
    category_codebook,
    feature_lists,
    numeric_bounds,
    reference_applicant,
    split_features_target,
)
from credit_risk.models import build_pipeline, positive_class_probability
from credit_risk.plots import (
    save_calibration,
    save_confusion_matrices,
    save_importance,
    save_model_comparison,
    save_policy_costs,
    save_precision_recall_curves,
    save_roc_curves,
    save_score_distribution,
    save_segment_rates,
    save_threshold_cost,
)
from credit_risk.profile import (
    categorical_default_rates,
    leading_segment_features,
    numeric_band_default_rates,
)
from credit_risk.train import cross_validate_models


def to_builtin(value):
    """Convert numpy and pandas scalars so ``json.dumps`` can write the report."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (str, int)) or value is None:
        return value
    if not isinstance(value, (list, dict)) and pd.isna(value):
        return None
    return value


def run_analysis(
    frame: pd.DataFrame,
    reports_dir: Path | None = None,
    model_path: Path | None = None,
    test_size: float = config.TEST_SIZE,
    n_splits: int = config.N_SPLITS,
    n_importance_repeats: int = config.N_IMPORTANCE_REPEATS,
    random_state: int = config.RANDOM_STATE,
) -> dict:
    """Compare models, tune the cutoff, and save tables, figures, and the scorer.

    The holdout is scored once. Model choice uses training-fold ROC-AUC. The
    cutoff minimizes Statlog cost on out-of-fold training scores.
    """
    reports = Path(reports_dir) if reports_dir is not None else config.REPORTS_DIR
    figures = reports / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    destination = Path(model_path) if model_path is not None else config.MODEL_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)

    features, target = split_features_target(frame)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )

    comparison = cross_validate_models(
        train_x, train_y, n_splits=n_splits, random_state=random_state
    )
    selected = comparison.best_model
    safe_numeric, safe_categorical = feature_lists(config.PROTECTED_FEATURES)
    reduced = cross_validate_models(
        train_x,
        train_y,
        model_names=(selected,),
        n_splits=n_splits,
        random_state=random_state,
        numeric_features=safe_numeric,
        categorical_features=safe_categorical,
    )

    sweeps = []
    thresholds: dict[str, float] = {}
    oof_rows = []
    for name in config.MODEL_NAMES:
        sweep = threshold_sweep(train_y.to_numpy(), comparison.oof_proba[name])
        sweep.insert(0, "model", name)
        sweeps.append(sweep)
        cutoff = choose_threshold(sweep)
        thresholds[name] = cutoff
        picked = row_at_threshold(sweep, cutoff)
        oof_rows.append(
            {
                "model": name,
                "threshold": cutoff,
                "expected_cost": float(picked["expected_cost"]),
                "expected_cost_per_applicant": float(picked["expected_cost_per_applicant"]),
                "approval_rate": float(picked["approval_rate"]),
                "recall_bad": float(picked["recall_bad"]),
                "precision_bad": float(picked["precision_bad"]),
            }
        )
    sweep_all = pd.concat(sweeps, ignore_index=True)
    oof_table = pd.DataFrame(oof_rows)

    fitted = {}
    test_probabilities = {}
    test_rows = []
    for name in config.MODEL_NAMES:
        pipeline = build_pipeline(name)
        pipeline.fit(train_x, train_y)
        fitted[name] = pipeline
        scores = positive_class_probability(pipeline, test_x)
        test_probabilities[name] = scores
        metrics = score_at_threshold(test_y.to_numpy(), scores, thresholds[name])
        cv_row = comparison.summary.loc[comparison.summary["model"] == name].iloc[0]
        metrics.update(
            {
                "model": name,
                "display_name": config.DISPLAY_NAMES[name],
                "selected": name == selected,
                "roc_auc_cv_mean": float(cv_row["roc_auc_mean"]),
                "roc_auc_cv_std": float(cv_row["roc_auc_std"]),
                "average_precision_cv_mean": float(cv_row["average_precision_mean"]),
                "average_precision_cv_std": float(cv_row["average_precision_std"]),
            }
        )
        test_rows.append(metrics)
    test_table = pd.DataFrame(test_rows)

    importance = permutation_importance_table(
        fitted[selected],
        test_x,
        test_y,
        n_repeats=n_importance_repeats,
        random_state=random_state,
    )
    sensitivity = cost_ratio_sensitivity(train_y.to_numpy(), comparison.oof_proba[selected])
    train_baselines = baseline_costs(train_y)
    test_baselines = baseline_costs(test_y)
    cat_rates = categorical_default_rates(features, target)
    num_rates = numeric_band_default_rates(features, target)
    top_categorical, top_numeric = leading_segment_features(importance)

    full_folds = comparison.fold_metrics.loc[comparison.fold_metrics["model"] == selected]
    full_folds = full_folds.sort_values("fold")
    reduced_folds = reduced.fold_metrics.sort_values("fold")
    deltas = full_folds["roc_auc"].to_numpy() - reduced_folds["roc_auc"].to_numpy()
    fairness = {
        "dropped_features": list(config.PROTECTED_FEATURES),
        "cv_roc_auc_with_protected": float(full_folds["roc_auc"].mean()),
        "cv_roc_auc_without_protected": float(reduced_folds["roc_auc"].mean()),
        "mean_auc_change_when_dropped": float(-deltas.mean()),
        "fold_auc_with_minus_without": [float(value) for value in deltas],
    }

    oof_leader = min(
        oof_table.to_dict(orient="records"),
        key=lambda row: (row["expected_cost"], row["model"]),
    )["model"]

    save_model_comparison(comparison.summary, figures / "model_comparison.png")
    save_roc_curves(test_y, test_probabilities, figures / "roc_curves.png")
    save_precision_recall_curves(test_y, test_probabilities, figures / "precision_recall.png")
    save_confusion_matrices(
        test_y,
        test_probabilities[selected],
        thresholds[selected],
        figures / "confusion_matrices.png",
    )
    save_threshold_cost(
        sweep_all.loc[sweep_all["model"] == selected],
        thresholds[selected],
        train_baselines,
        figures / "threshold_cost.png",
    )
    save_policy_costs(test_table, test_baselines, selected, figures / "policy_cost.png")
    save_importance(importance, figures / "permutation_importance.png")
    save_score_distribution(
        test_y,
        test_probabilities[selected],
        thresholds[selected],
        figures / "score_distribution.png",
    )
    save_calibration(test_y, test_probabilities[selected], figures / "calibration.png")
    save_segment_rates(
        cat_rates,
        num_rates,
        top_categorical,
        top_numeric,
        base_rate=float(target.mean()),
        path=figures / "segment_default_rates.png",
    )

    selected_test = test_table.loc[test_table["model"] == selected].iloc[0].to_dict()
    codebook = category_codebook(features)
    artifact = {
        "artifact_version": config.ARTIFACT_VERSION,
        "model_name": selected,
        "display_name": config.DISPLAY_NAMES[selected],
        "pipeline": fitted[selected],
        "threshold": thresholds[selected],
        "review_band": config.REVIEW_BAND,
        "feature_columns": list(config.FEATURE_COLUMNS),
        "numeric_columns": list(config.NUMERIC_FEATURES),
        "categorical_columns": list(config.CATEGORICAL_FEATURES),
        "category_levels": codebook,
        "numeric_bounds": numeric_bounds(features),
        "reference_applicant": reference_applicant(train_x),
        "cost_false_approval": config.COST_FALSE_APPROVAL,
        "cost_false_denial": config.COST_FALSE_DENIAL,
        "illustrative_denial_usd": config.ILLUSTRATIVE_DENIAL_USD,
        "illustrative_false_approval_usd": config.COST_FALSE_APPROVAL
        * config.ILLUSTRATIVE_DENIAL_USD,
        "positive_label": config.POSITIVE_LABEL,
        "protected_features": list(config.PROTECTED_FEATURES),
        "test_snapshot": {
            key: selected_test[key]
            for key in (
                "roc_auc",
                "average_precision",
                "brier",
                "threshold",
                "precision_bad",
                "recall_bad",
                "approval_rate",
                "default_rate_among_approved",
                "expected_cost_per_applicant",
                "illustrative_cost_usd",
                "illustrative_cost_per_applicant_usd",
            )
        },
        "top_features": importance.head(8).to_dict(orient="records"),
        "fairness": fairness,
    }
    joblib.dump(artifact, destination)

    payload = {
        "dataset": {
            "name": config.DATASET_NAME,
            "citation": config.DATASET_CITATION,
            "n": int(len(features)),
            "n_train": int(len(train_x)),
            "n_test": int(len(test_x)),
            "default_rate": float(target.mean()),
            "missing_numeric": int(features[config.NUMERIC_FEATURES].isna().sum().sum()),
            "missing_categorical": int(features[config.CATEGORICAL_FEATURES].isna().sum().sum()),
        },
        "policy": {
            "positive_class": "default (dataset label 'bad')",
            "cost_false_approval": config.COST_FALSE_APPROVAL,
            "cost_false_denial": config.COST_FALSE_DENIAL,
            "illustrative_denial_usd": config.ILLUSTRATIVE_DENIAL_USD,
            "illustrative_false_approval_usd": config.COST_FALSE_APPROVAL
            * config.ILLUSTRATIVE_DENIAL_USD,
            "selection_rule": (
                "Highest mean stratified cross-validated ROC-AUC on the training set. "
                "Ties break toward the smaller fold standard deviation, then the model name."
            ),
            "threshold_rule": (
                "On training out-of-fold scores, choose the cutoff with the lowest "
                "Statlog expected cost. Equal costs keep the higher cutoff."
            ),
        },
        "cv_summary": comparison.summary.to_dict(orient="records"),
        "oof_operating_points": oof_table.to_dict(orient="records"),
        "selected_model": selected,
        "oof_cost_leader": oof_leader,
        "selected_threshold": thresholds[selected],
        "test_metrics": test_table.to_dict(orient="records"),
        "test_baselines": test_baselines,
        "train_baselines": train_baselines,
        "cost_sensitivity": sensitivity.to_dict(orient="records"),
        "fairness": fairness,
        "top_features": importance.head(10).to_dict(orient="records"),
        "segment_features": {"categorical": top_categorical, "numeric": top_numeric},
        "design": {
            "test_size": test_size,
            "n_splits": n_splits,
            "n_importance_repeats": n_importance_repeats,
            "random_state": random_state,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    payload = to_builtin(payload)

    comparison.summary.to_csv(reports / "model_comparison.csv", index=False)
    comparison.fold_metrics.to_csv(reports / "fold_metrics.csv", index=False)
    sweep_all.to_csv(reports / "threshold_sweep.csv", index=False)
    test_table.to_csv(reports / "test_metrics.csv", index=False)
    oof_table.to_csv(reports / "oof_operating_points.csv", index=False)
    importance.to_csv(reports / "permutation_importance.csv", index=False)
    sensitivity.to_csv(reports / "cost_sensitivity.csv", index=False)
    cat_rates.to_csv(reports / "categorical_default_rates.csv", index=False)
    num_rates.to_csv(reports / "numeric_band_default_rates.csv", index=False)
    (reports / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> dict:
    """Load the credit file and write the full study to ``reports/``."""
    parser = argparse.ArgumentParser(description="Train and evaluate the German-credit risk study.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload credit-g from OpenML before training.",
    )
    parser.add_argument("--cache", type=Path, default=config.CACHE_PATH)
    parser.add_argument("--reports", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--splits", type=int, default=config.N_SPLITS)
    parser.add_argument("--importance-repeats", type=int, default=config.N_IMPORTANCE_REPEATS)
    args = parser.parse_args(argv)
    frame = load_german_credit(cache_path=args.cache, refresh=args.refresh)
    payload = run_analysis(
        frame,
        reports_dir=args.reports,
        model_path=args.model,
        n_splits=args.splits,
        n_importance_repeats=args.importance_repeats,
    )
    selected = payload["selected_model"]
    threshold = payload["selected_threshold"]
    holdout = next(row for row in payload["test_metrics"] if row["model"] == selected)
    print(
        f"Selected {config.DISPLAY_NAMES[selected]} at cutoff {threshold:.2f}. "
        f"Holdout ROC-AUC {holdout['roc_auc']:.3f}, "
        f"illustrative cost ${holdout['illustrative_cost_usd']:,.0f}."
    )
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()
