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
from credit_risk.bootstrap import (
    BOOTSTRAP_METHOD,
    auc_difference_sentence,
    bootstrap_scores,
)
from credit_risk.curves import constant_brier, policy_curve
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
from credit_risk.fairness import fairness_report
from credit_risk.features import (
    category_codebook,
    feature_lists,
    numeric_bounds,
    reference_applicant,
    split_features_target,
)
from credit_risk.model_card import render_model_card
from credit_risk.models import build_pipeline, positive_class_probability
from credit_risk.plots import (
    save_auc_intervals,
    save_calibration,
    save_confusion_matrices,
    save_importance,
    save_model_comparison,
    save_policy_costs,
    save_precision_recall_curves,
    save_profit_curve,
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
from credit_risk.train import (
    LOGISTIC_OPTIMISM_NOTE,
    LOGISTIC_SELECTION_RULE,
    cross_validate_models,
    select_logistic_c,
)


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


def _boot_value(table: pd.DataFrame, split: str, comparison: str, model: str, metric: str):
    match = table.loc[
        (table["split"] == split)
        & (table["comparison"] == comparison)
        & (table["model"] == model)
        & (table["metric"] == metric)
    ]
    if match.empty:
        raise KeyError(f"No bootstrap row for {split} {comparison} {model} {metric}.")
    return match.iloc[0]


def run_analysis(
    frame: pd.DataFrame,
    reports_dir: Path | None = None,
    model_path: Path | None = None,
    test_size: float = config.TEST_SIZE,
    n_splits: int = config.N_SPLITS,
    n_importance_repeats: int = config.N_IMPORTANCE_REPEATS,
    n_bootstrap: int = config.N_BOOTSTRAP,
    random_state: int = config.RANDOM_STATE,
    model_card_path: Path | None = None,
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

    chosen_c, c_grid, baseline_oof = select_logistic_c(
        train_x,
        train_y,
        n_splits=n_splits,
        random_state=random_state,
    )
    if math.isclose(chosen_c, 1.0):
        baseline_test_scores = test_probabilities["logistic_regression"]
    else:
        baseline_pipeline = build_pipeline("logistic_regression", logistic_c=chosen_c)
        baseline_pipeline.fit(train_x, train_y)
        baseline_test_scores = positive_class_probability(baseline_pipeline, test_x)
    baseline_name = config.TUNED_BASELINE_NAME
    test_probabilities[baseline_name] = baseline_test_scores
    baseline_sweep = threshold_sweep(train_y.to_numpy(), baseline_oof)
    baseline_threshold = choose_threshold(baseline_sweep)
    baseline_picked = row_at_threshold(baseline_sweep, baseline_threshold)
    grid_match = c_grid.loc[np.isclose(c_grid["C"].to_numpy(), chosen_c)]
    grid_row = grid_match.iloc[0]
    baseline_holdout = score_at_threshold(
        test_y.to_numpy(), baseline_test_scores, baseline_threshold
    )
    baseline_holdout.update(
        {
            "model": baseline_name,
            "display_name": config.DISPLAY_NAMES[baseline_name],
            "selected": False,
            "roc_auc_cv_mean": float(grid_row["roc_auc_mean"]),
            "roc_auc_cv_std": float(grid_row["roc_auc_std"]),
            "average_precision_cv_mean": float(grid_row["average_precision_mean"]),
            "average_precision_cv_std": float(grid_row["average_precision_std"]),
        }
    )
    test_table = pd.concat([test_table, pd.DataFrame([baseline_holdout])], ignore_index=True)
    oof_table = pd.concat(
        [
            oof_table,
            pd.DataFrame(
                [
                    {
                        "model": baseline_name,
                        "threshold": baseline_threshold,
                        "expected_cost": float(baseline_picked["expected_cost"]),
                        "expected_cost_per_applicant": float(
                            baseline_picked["expected_cost_per_applicant"]
                        ),
                        "approval_rate": float(baseline_picked["approval_rate"]),
                        "recall_bad": float(baseline_picked["recall_bad"]),
                        "precision_bad": float(baseline_picked["precision_bad"]),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    score_bank = {name: comparison.oof_proba[name] for name in config.MODEL_NAMES}
    score_bank[baseline_name] = baseline_oof
    cutoff_bank = dict(thresholds)
    cutoff_bank[baseline_name] = baseline_threshold
    oof_boot = bootstrap_scores(
        train_y.to_numpy(),
        score_bank,
        n_boot=n_bootstrap,
        random_state=random_state,
        alpha=config.BOOTSTRAP_ALPHA,
        baseline=baseline_name,
        thresholds=cutoff_bank,
    )
    oof_boot.insert(0, "split", "out_of_fold")
    holdout_bank = {name: test_probabilities[name] for name in score_bank}
    holdout_boot = bootstrap_scores(
        test_y.to_numpy(),
        holdout_bank,
        n_boot=n_bootstrap,
        random_state=random_state,
        alpha=config.BOOTSTRAP_ALPHA,
        baseline=baseline_name,
        thresholds=cutoff_bank,
    )
    holdout_boot.insert(0, "split", "holdout")
    boot_table = pd.concat([oof_boot, holdout_boot], ignore_index=True)

    comparison_rows = []
    for name in list(config.MODEL_NAMES) + [baseline_name]:
        auc_row = _boot_value(boot_table, "out_of_fold", "level", name, "roc_auc")
        brier_row = _boot_value(boot_table, "out_of_fold", "level", name, "brier")
        holdout_row = test_table.loc[test_table["model"] == name].iloc[0]
        if name == baseline_name:
            role = "Tuned baseline"
            cv_mean = float(grid_row["roc_auc_mean"])
            cv_std = float(grid_row["roc_auc_std"])
            cutoff = baseline_threshold
        else:
            role = "Selected model" if name == selected else "Candidate"
            cv_mean = float(holdout_row["roc_auc_cv_mean"])
            cv_std = float(holdout_row["roc_auc_cv_std"])
            cutoff = thresholds[name]
        comparison_rows.append(
            {
                "model": name,
                "display_name": config.DISPLAY_NAMES[name],
                "role": role,
                "in_selection_race": name in config.MODEL_NAMES,
                "roc_auc_cv_mean": cv_mean,
                "roc_auc_cv_std": cv_std,
                "oof_roc_auc": float(auc_row["estimate"]),
                "oof_roc_auc_ci_low": float(auc_row["ci_low"]),
                "oof_roc_auc_ci_high": float(auc_row["ci_high"]),
                "oof_brier": float(brier_row["estimate"]),
                "oof_brier_ci_low": float(brier_row["ci_low"]),
                "oof_brier_ci_high": float(brier_row["ci_high"]),
                "threshold": float(cutoff),
                "holdout_roc_auc": float(holdout_row["roc_auc"]),
                "holdout_brier": float(holdout_row["brier"]),
                "holdout_illustrative_cost_usd": float(holdout_row["illustrative_cost_usd"]),
                "holdout_approval_rate": float(holdout_row["approval_rate"]),
            }
        )
    comparison_table = pd.DataFrame(comparison_rows)
    auc_gap = _boot_value(boot_table, "out_of_fold", "difference", selected, "roc_auc")
    brier_gap = _boot_value(boot_table, "out_of_fold", "difference", selected, "brier")
    cost_gap = _boot_value(boot_table, "out_of_fold", "difference", selected, "cost_per_applicant")
    versus_sentence = auc_difference_sentence(
        config.DISPLAY_NAMES[selected],
        config.DISPLAY_NAMES[baseline_name],
        float(auc_gap["estimate"]),
        float(auc_gap["ci_low"]),
        float(auc_gap["ci_high"]),
    )
    best_vs_baseline = {
        "model": selected,
        "baseline": baseline_name,
        "sentence": versus_sentence,
        "oof_roc_auc_difference": float(auc_gap["estimate"]),
        "oof_roc_auc_ci_low": float(auc_gap["ci_low"]),
        "oof_roc_auc_ci_high": float(auc_gap["ci_high"]),
        "oof_roc_auc_ci_includes_zero": bool(auc_gap["includes_zero"]),
        "oof_brier_difference": float(brier_gap["estimate"]),
        "oof_brier_ci_low": float(brier_gap["ci_low"]),
        "oof_brier_ci_high": float(brier_gap["ci_high"]),
        "oof_cost_per_applicant_difference": float(cost_gap["estimate"]),
        "oof_cost_ci_low": float(cost_gap["ci_low"]),
        "oof_cost_ci_high": float(cost_gap["ci_high"]),
        "holdout_cost_usd_model": float(
            test_table.loc[test_table["model"] == selected, "illustrative_cost_usd"].iloc[0]
        ),
        "holdout_cost_usd_baseline": float(baseline_holdout["illustrative_cost_usd"]),
    }

    oof_fair = fairness_report(
        train_y.to_numpy(),
        comparison.oof_proba[selected],
        train_x["age"],
        train_x["personal_status"],
        thresholds[selected],
        split_label="training out-of-fold scores",
    )
    holdout_fair = fairness_report(
        test_y.to_numpy(),
        test_probabilities[selected],
        test_x["age"],
        test_x["personal_status"],
        thresholds[selected],
        split_label="locked holdout",
    )
    group_fairness = {
        "age_cut": config.AGE_GROUP_CUT,
        "sex_definition": (
            "personal_status values that start with 'female' or 'male'; "
            "marital status inside the label is collapsed."
        ),
        "oof": {
            "n": oof_fair["n"],
            "threshold": oof_fair["threshold"],
            "groups": oof_fair["rates"].to_dict(orient="records"),
            "gaps": oof_fair["gaps"].to_dict(orient="records"),
            "note": oof_fair["note"],
        },
        "holdout": {
            "n": holdout_fair["n"],
            "threshold": holdout_fair["threshold"],
            "groups": holdout_fair["rates"].to_dict(orient="records"),
            "gaps": holdout_fair["gaps"].to_dict(orient="records"),
            "note": holdout_fair["note"],
        },
    }
    calibration = {
        "holdout_brier": float(test_table.loc[test_table["model"] == selected, "brier"].iloc[0]),
        "holdout_brier_constant": constant_brier(test_y),
        "oof_brier": float(
            _boot_value(boot_table, "out_of_fold", "level", selected, "brier")["estimate"]
        ),
        "oof_brier_constant": constant_brier(train_y),
    }
    profit = policy_curve(
        train_y.to_numpy(),
        comparison.oof_proba[selected],
        config.COST_FALSE_APPROVAL,
        config.COST_FALSE_DENIAL,
    )

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

    race_oof = oof_table.loc[oof_table["model"].isin(config.MODEL_NAMES)]
    oof_leader = min(
        race_oof.to_dict(orient="records"),
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
    save_profit_curve(profit, thresholds[selected], figures / "profit_curve.png")
    auc_intervals = boot_table.loc[
        (boot_table["split"] == "out_of_fold")
        & (boot_table["comparison"] == "level")
        & (boot_table["metric"] == "roc_auc")
    ].copy()
    auc_order = list(config.MODEL_NAMES) + [baseline_name]
    auc_intervals["model"] = pd.Categorical(
        auc_intervals["model"], categories=auc_order, ordered=True
    )
    auc_intervals = auc_intervals.sort_values("model")
    save_auc_intervals(auc_intervals, figures / "auc_intervals.png")
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
        "comparison_table": comparison_table.to_dict(orient="records"),
        "comparison_sentence": versus_sentence,
        "desk": {
            "oof_y": train_y.astype(int).tolist(),
            "oof_scores": comparison.oof_proba[selected].tolist(),
            "holdout_y": test_y.astype(int).tolist(),
            "holdout_scores": test_probabilities[selected].tolist(),
            "holdout_brier": calibration["holdout_brier"],
            "holdout_brier_constant": calibration["holdout_brier_constant"],
            "oof_brier": calibration["oof_brier"],
            "oof_brier_constant": calibration["oof_brier_constant"],
        },
        "group_fairness": {
            "age_cut": group_fairness["age_cut"],
            "sex_definition": group_fairness["sex_definition"],
            "oof_note": oof_fair["note"],
            "holdout_note": holdout_fair["note"],
            "oof_groups": oof_fair["rates"].to_dict(orient="records"),
            "oof_gaps": oof_fair["gaps"].to_dict(orient="records"),
            "holdout_groups": holdout_fair["rates"].to_dict(orient="records"),
            "holdout_gaps": holdout_fair["gaps"].to_dict(orient="records"),
        },
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
        "tuned_baseline": {
            "model": baseline_name,
            "display_name": config.DISPLAY_NAMES[baseline_name],
            "family": config.BASELINE_FAMILY,
            "C": chosen_c,
            "C_grid": [float(value) for value in config.LOGISTIC_C_GRID],
            "C_results": c_grid.to_dict(orient="records"),
            "selection_rule": LOGISTIC_SELECTION_RULE,
            "optimism_note": LOGISTIC_OPTIMISM_NOTE,
            "cv_roc_auc_mean": float(grid_row["roc_auc_mean"]),
            "cv_roc_auc_std": float(grid_row["roc_auc_std"]),
            "oof_threshold": baseline_threshold,
            "oof_expected_cost": float(baseline_picked["expected_cost"]),
            "holdout": {
                key: baseline_holdout[key]
                for key in (
                    "roc_auc",
                    "average_precision",
                    "brier",
                    "threshold",
                    "approval_rate",
                    "illustrative_cost_usd",
                    "expected_cost",
                )
            },
        },
        "best_vs_baseline": best_vs_baseline,
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "n_boot": n_bootstrap,
            "alpha": config.BOOTSTRAP_ALPHA,
            "random_state": random_state,
        },
        "comparison": comparison_table.to_dict(orient="records"),
        "group_fairness": group_fairness,
        "calibration": calibration,
        "top_features": importance.head(10).to_dict(orient="records"),
        "segment_features": {"categorical": top_categorical, "numeric": top_numeric},
        "design": {
            "test_size": test_size,
            "n_splits": n_splits,
            "n_importance_repeats": n_importance_repeats,
            "n_bootstrap": n_bootstrap,
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

    comparison_table.to_csv(reports / "model_comparison.csv", index=False)
    comparison.fold_metrics.to_csv(reports / "fold_metrics.csv", index=False)
    boot_table.to_csv(reports / "bootstrap_metrics.csv", index=False)
    c_grid.to_csv(reports / "logistic_c_grid.csv", index=False)
    group_rows = pd.concat(
        [
            oof_fair["rates"].assign(split="out_of_fold"),
            holdout_fair["rates"].assign(split="holdout"),
        ],
        ignore_index=True,
    )
    gap_rows = pd.concat(
        [
            oof_fair["gaps"].assign(split="out_of_fold"),
            holdout_fair["gaps"].assign(split="holdout"),
        ],
        ignore_index=True,
    )
    group_rows.to_csv(reports / "fairness_groups.csv", index=False)
    gap_rows.to_csv(reports / "fairness_gaps.csv", index=False)
    sweep_all.to_csv(reports / "threshold_sweep.csv", index=False)
    test_table.to_csv(reports / "test_metrics.csv", index=False)
    oof_table.to_csv(reports / "oof_operating_points.csv", index=False)
    importance.to_csv(reports / "permutation_importance.csv", index=False)
    sensitivity.to_csv(reports / "cost_sensitivity.csv", index=False)
    cat_rates.to_csv(reports / "categorical_default_rates.csv", index=False)
    num_rates.to_csv(reports / "numeric_band_default_rates.csv", index=False)
    (reports / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    card = render_model_card(payload)
    card_in_reports = reports / "MODEL_CARD.md"
    card_in_reports.write_text(card, encoding="utf-8")
    if model_card_path is not None:
        destination_card = Path(model_card_path)
        if destination_card.resolve() != card_in_reports.resolve():
            destination_card.parent.mkdir(parents=True, exist_ok=True)
            destination_card.write_text(card, encoding="utf-8")
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
    parser.add_argument("--bootstrap", type=int, default=config.N_BOOTSTRAP)
    parser.add_argument(
        "--model-card",
        type=Path,
        default=None,
        help="Also write MODEL_CARD.md to this path.",
    )
    args = parser.parse_args(argv)
    frame = load_german_credit(cache_path=args.cache, refresh=args.refresh)
    card_path = args.model_card
    if card_path is None and Path(args.reports).resolve() == config.REPORTS_DIR.resolve():
        card_path = config.PROJECT_ROOT / "MODEL_CARD.md"
    payload = run_analysis(
        frame,
        reports_dir=args.reports,
        model_path=args.model,
        n_splits=args.splits,
        n_importance_repeats=args.importance_repeats,
        n_bootstrap=args.bootstrap,
        model_card_path=card_path,
    )
    selected = payload["selected_model"]
    threshold = payload["selected_threshold"]
    holdout = next(row for row in payload["test_metrics"] if row["model"] == selected)
    print(
        f"Selected {config.DISPLAY_NAMES[selected]} at cutoff {threshold:.2f}. "
        f"Holdout ROC-AUC {holdout['roc_auc']:.3f}, "
        f"illustrative cost ${holdout['illustrative_cost_usd']:,.0f}."
    )
    print(payload["best_vs_baseline"]["sentence"])
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()
