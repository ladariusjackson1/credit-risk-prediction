"""Model card rendered from a study payload, so the figures cannot drift."""

from __future__ import annotations

from collections.abc import Mapping


def _need(mapping: Mapping, key: str):
    if key not in mapping:
        raise KeyError(f"Model card payload is missing '{key}'.")
    return mapping[key]


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "not estimated"
    return f"{float(value):.{digits}f}"


def _pct(value: object, digits: int = 1) -> str:
    if value is None:
        return "not estimated"
    return f"{float(value):.{digits}%}"


def _dollars(value: object) -> str:
    if value is None:
        return "not estimated"
    return f"${float(value):,.0f}"


def _ci(low: object, high: object, digits: int = 3) -> str:
    return f"{_fmt(low, digits)} to {_fmt(high, digits)}"


def _comparison_table(rows: list[Mapping]) -> str:
    header = (
        "| Model | Role | CV ROC-AUC | OOF ROC-AUC (95% CI) | OOF Brier (95% CI) | "
        "Cutoff | Holdout ROC-AUC | Holdout Brier | Holdout illustrative cost |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    body = []
    for row in rows:
        cv = f"{_fmt(row['roc_auc_cv_mean'])} ± {_fmt(row['roc_auc_cv_std'])}"
        oof_auc = f"{_fmt(row['oof_roc_auc'])} ({_ci(row['oof_roc_auc_ci_low'], row['oof_roc_auc_ci_high'])})"
        oof_brier = (
            f"{_fmt(row['oof_brier'])} ({_ci(row['oof_brier_ci_low'], row['oof_brier_ci_high'])})"
        )
        body.append(
            "| {name} | {role} | {cv} | {oof_auc} | {oof_brier} | {cut} | {hold_auc} | "
            "{hold_brier} | {cost} |".format(
                name=row["display_name"],
                role=row["role"],
                cv=cv,
                oof_auc=oof_auc,
                oof_brier=oof_brier,
                cut=f"{float(row['threshold']):.2f}",
                hold_auc=_fmt(row["holdout_roc_auc"]),
                hold_brier=_fmt(row["holdout_brier"]),
                cost=_dollars(row["holdout_illustrative_cost_usd"]),
            )
        )
    return "\n".join([header, *body])


def render_model_card(payload: Mapping) -> str:
    """Write the model card markdown from ``payload`` produced by the study."""
    dataset = _need(payload, "dataset")
    policy = _need(payload, "policy")
    baseline = _need(payload, "tuned_baseline")
    versus = _need(payload, "best_vs_baseline")
    bootstrap = _need(payload, "bootstrap")
    comparison = list(_need(payload, "comparison"))
    groups = _need(payload, "group_fairness")
    dropped = _need(payload, "fairness")
    calibration = _need(payload, "calibration")
    design = _need(payload, "design")
    environment = _need(payload, "environment")
    top_features = list(_need(payload, "top_features"))
    selected = str(_need(payload, "selected_model"))
    selected_row = next(row for row in comparison if row["model"] == selected)
    threshold = float(_need(payload, "selected_threshold"))
    false_approval = float(policy["cost_false_approval"])
    false_denial = float(policy["cost_false_denial"])
    top_line = "No permutation-importance row was stored."
    if top_features:
        leader = top_features[0]
        top_line = (
            f"The largest holdout permutation drop is `{leader['feature']}` "
            f"at {_fmt(leader['importance_mean'])} ROC-AUC "
            f"(repeat spread {_fmt(leader['importance_std'])})."
        )
    grid = ", ".join(f"{float(value):g}" for value in baseline["C_grid"])
    dropped_bits = [f"`{name}`" for name in dropped["dropped_features"]]
    if len(dropped_bits) == 1:
        dropped_names = dropped_bits[0]
    elif len(dropped_bits) == 2:
        dropped_names = f"{dropped_bits[0]} and {dropped_bits[1]}"
    else:
        dropped_names = ", ".join(dropped_bits[:-1]) + f", and {dropped_bits[-1]}"
    interval = f"{1.0 - float(bootstrap['alpha']):.0%}"

    return f"""# Model card: German credit desk score

## Model details

This card describes the scorer written by `python -m credit_risk` and saved at `models/credit_risk_model.joblib`.

- **Shipped model:** {selected_row["display_name"]} (`{selected}`).
- **Decision:** approve when the estimated probability of default is below {threshold:.2f}; otherwise deny.
- **Positive class:** {policy["positive_class"]}.
- **Training rows:** {int(dataset["n_train"])}. **Locked holdout:** {int(dataset["n_test"])}. Stratified split, `random_state={int(design["random_state"])}`, test size {float(design["test_size"]):.0%}.
- **Cross-validation:** {int(design["n_splits"])} stratified folds on the training rows.
- **Selection rule:** {policy["selection_rule"]}
- **Cutoff rule:** {policy["threshold_rule"]}
- **Environment:** Python {environment["python"]}, scikit-learn {environment["scikit_learn"]}, pandas {environment["pandas"]}, NumPy {environment["numpy"]}.

The shipped model is the winner of a fixed-hyperparameter race among logistic regression, a random forest, and histogram gradient boosting. A tuned logistic baseline is reported beside it and is not allowed to replace the shipped model after the holdout is opened.

## Intended use

The score is a portfolio demonstration of a cost-sensitive credit decision on a public 1994 German installment file. It is meant to rank applicants from that file and to show how a published loss ratio moves the approval cutoff.

A suitable use is a credit-risk, analytics, or model-risk discussion: what the cutoff costs, how uncertain the model ranking is, and where approval rates differ by age and by the file's sex label.

## Out of scope

- A live lending decision, a pricing engine, a credit limit, or an adverse-action notice.
- Expected loss, capital, reserves, or a dollar forecast. The dollars in this project translate the Statlog cost matrix at {_dollars(policy["illustrative_denial_usd"])} per cost unit. A false approval is {false_approval:g} units ({_dollars(policy["illustrative_false_approval_usd"])}). A false denial is {false_denial:g} unit ({_dollars(policy["illustrative_denial_usd"])}).
- Any population other than this file. It is not a 2026 book and it is not a U.S. application population.
- Fair-lending compliance, disparate-impact testing, or a decision about a protected class.

## Data

- **Source:** {dataset["name"]}.
- **Citation:** {dataset["citation"]}.
- **Rows:** {int(dataset["n"])}. **Default rate:** {_pct(dataset["default_rate"])} (the dataset label `bad`).
- **Missing numeric values:** {int(dataset["missing_numeric"])}. **Missing categorical values:** {int(dataset["missing_categorical"])}.
- **Inputs:** the 20 Statlog attributes. Numeric fields are median-imputed and scaled. Categorical fields are mode-imputed and one-hot encoded. Unknown categories at score time become zeros.
- **Sensitive inputs kept in the benchmark:** `personal_status` (sex and marital status together) and `foreign_worker` (nationality). They are in the public file, so the benchmark can be reproduced. They should not be inputs to a live U.S. decision.

## Metrics

Bootstrap method: {bootstrap["method"]} Replicates: {int(bootstrap["n_boot"])}. Interval: {interval} percentile. Seed: {int(bootstrap["random_state"])}.

Tuned baseline: {baseline["display_name"]}, logistic regression with C = {_fmt(baseline["C"], 1)} chosen from {{{grid}}}. {baseline["selection_rule"]} {baseline["optimism_note"]}

{versus["sentence"]}

The out-of-fold Brier difference, shipped model minus baseline, is {_fmt(versus["oof_brier_difference"])} (95% interval {_ci(versus["oof_brier_ci_low"], versus["oof_brier_ci_high"])}). A negative Brier difference means the shipped scores are closer to the observed outcomes. The out-of-fold cost-per-applicant difference at each model's own cutoff is {_fmt(versus["oof_cost_per_applicant_difference"])} (95% interval {_ci(versus["oof_cost_ci_low"], versus["oof_cost_ci_high"])}). A negative cost difference means the shipped policy costs less on these out-of-fold scores. Holdout illustrative cost is {_dollars(versus["holdout_cost_usd_model"])} for the shipped model and {_dollars(versus["holdout_cost_usd_baseline"])} for the tuned baseline. The holdout was not used to pick the model or the cutoff.

{_comparison_table(comparison)}

CV ROC-AUC is the mean across training folds, with the fold standard deviation. OOF metrics are computed on the stacked out-of-fold scores. The interval is the bootstrap described above. Holdout columns are a single locked look.

### Calibration

Class weights are balanced, so a score of 0.15 is not a claim that the applicant has a 15% chance of default. On the holdout, the shipped model's Brier score is {_fmt(calibration["holdout_brier"])}. Predicting the holdout's own default rate for every applicant scores {_fmt(calibration["holdout_brier_constant"])}. On the training out-of-fold scores the Brier score is {_fmt(calibration["oof_brier"])}, against {_fmt(calibration["oof_brier_constant"])} for the constant out-of-fold default rate. Use the score to rank and to set a cutoff. Do not book a loss reserve from the raw score. The reliability diagram is `reports/figures/calibration.png`.

## Fairness

Sex in this check is the first word of `personal_status`. {groups["sex_definition"]} Age bands are split at {float(groups["age_cut"]):g}.

### Training out-of-fold scores

{groups["oof"]["note"]}

### Locked holdout

{groups["holdout"]["note"]}

Dropping {dropped_names} and repeating the same folds changes the shipped model's mean ROC-AUC from {_fmt(dropped["cv_roc_auc_with_protected"])} to {_fmt(dropped["cv_roc_auc_without_protected"])} (change when dropped: {_fmt(dropped["mean_auc_change_when_dropped"])}). Ranking quality on this file does not depend on those two columns in any way that this sample can pin down. A policy build should refit without them. The saved scorer still asks for them so a score matches the benchmark that was trained.

## Factors

{top_line} Permutation importance shuffles one original column at a time on the holdout. The codebook is not modern American English: in this file, having no checking account is the low-risk checking status. Read the empirical rates in `reports/categorical_default_rates.csv` before reading the English gloss.

Local explanations in the app and in a batch file are one-field swaps against the typical training applicant (median numerics, most common category). They are not an additive attribution, and they are not SHAP.

## Limits

- One thousand rows from 1994. A few defaults moving on the 200-row holdout can reorder model costs. That reorder is reported and is not used to change the shipped model.
- The bootstrap holds the fitted scores fixed. It is not a nested refit, and it does not re-choose the logistic penalty.
- The logistic penalty was chosen on the same folds that are reported, over a three-point grid. That can flatter the baseline slightly.
- Balanced class weights and the cutoff do different jobs. Weights change the fit. The cutoff is the desk's decision.
- Illustrative dollars are a translation of the published matrix, not a loss model fit from the file.
- `personal_status` and `foreign_worker` remain in the benchmark inputs.
- Scores were not calibrated to a live portfolio's base rate.

## Ethical considerations

A denial from this score is not an adverse-action reason. The batch explanation says which fields move the score relative to a typical training row; it does not establish a legally sufficient reason. Sex, marital status, age, and nationality are present in the file. The fairness section describes gaps. It does not clear the model for use, and a gap near zero would not clear it either.
"""
