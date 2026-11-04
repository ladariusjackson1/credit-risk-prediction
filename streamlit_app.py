"""Desk tool for Streamlit Community Cloud.

Run from the repository root:

    streamlit run streamlit_app.py

Community Cloud uses that same command with the repository root as the working
directory. Scoring reads ``models/credit_risk_model.joblib``. It does not
download the dataset and it does not call OpenML.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import streamlit as st

from credit_risk.batch import batch_template, score_batch
from credit_risk.config import DISPLAY_NAMES
from credit_risk.curves import policy_curve
from credit_risk.evaluate import choose_threshold, row_at_threshold
from credit_risk.explain import explain_applicant
from credit_risk.plots import draw_auc_intervals, draw_calibration, draw_profit
from credit_risk.scoring import decision_narrative, load_artifact

LABELS = {
    "duration": "Duration (months)",
    "credit_amount": "Credit amount (DM)",
    "installment_commitment": "Installment rate (% of disposable income)",
    "residence_since": "Years at current residence",
    "age": "Age (years)",
    "existing_credits": "Existing credits at this bank",
    "num_dependents": "Dependents",
    "checking_status": "Checking account status",
    "credit_history": "Credit history",
    "purpose": "Purpose",
    "savings_status": "Savings account / bonds",
    "employment": "Present employment since",
    "personal_status": "Personal status and sex",
    "other_parties": "Other debtors / guarantors",
    "property_magnitude": "Property",
    "other_payment_plans": "Other installment plans",
    "housing": "Housing",
    "job": "Job",
    "own_telephone": "Telephone",
    "foreign_worker": "Foreign worker",
}

HELP = {
    "duration": "Length of the requested installment loan, in months.",
    "credit_amount": "Requested amount in the dataset's currency (Deutsche Marks).",
    "installment_commitment": "Installment size as a share of disposable income, coded 1 to 4.",
    "checking_status": "Checking-account balance. 'no checking' means the applicant has none.",
    "credit_history": "Status of earlier credits, using the Statlog codebook labels.",
    "savings_status": "Savings or bond holdings at this bank.",
    "employment": "How long the applicant has been with the current employer.",
    "personal_status": "Sex and marital status from the 1994 file. Not for a live US policy.",
    "foreign_worker": "Nationality flag from the public file. Not for a live US policy.",
    "purpose": "What the loan is for, in the dataset's purpose codes.",
    "other_parties": "Whether a guarantor or co-applicant is on the application.",
    "property_magnitude": "The most substantial collateral or property on file.",
    "other_payment_plans": "Installment plans the applicant already has outside this loan.",
    "housing": "Whether the applicant owns, rents, or lives rent-free.",
    "job": "Skill and employment category.",
    "own_telephone": "Whether a telephone is registered in the applicant's name.",
    "age": "Age in years.",
    "residence_since": "Years at the current address, coded 1 to 4.",
    "existing_credits": "Count of existing credits at this bank.",
    "num_dependents": "Number of people the applicant is liable to maintain.",
}

GROUPS = [
    (
        "Loan request",
        ["duration", "credit_amount", "purpose", "installment_commitment"],
        None,
    ),
    (
        "Bank relationship",
        [
            "checking_status",
            "savings_status",
            "credit_history",
            "existing_credits",
            "other_payment_plans",
            "other_parties",
        ],
        None,
    ),
    (
        "Stability",
        [
            "employment",
            "job",
            "property_magnitude",
            "housing",
            "residence_since",
            "age",
            "num_dependents",
            "own_telephone",
        ],
        None,
    ),
    (
        "Attributes that should not drive a live US decision",
        ["personal_status", "foreign_worker"],
        (
            "These two fields are in the benchmark file. personal_status encodes sex and "
            "marital status, and foreign_worker encodes nationality. The study measures "
            "what happens to ranking quality when they are removed. They are on this form "
            "only so a score matches the model that was trained."
        ),
    ),
]


@st.cache_resource(show_spinner=False)
def cached_artifact() -> dict:
    """Load the joblib bundle committed in ``models/``. Local disk only."""
    return load_artifact()


def _seed(artifact: dict) -> None:
    reference = artifact["reference_applicant"]
    for column in artifact["feature_columns"]:
        key = f"field_{column}"
        if key not in st.session_state:
            value = reference[column]
            if column in artifact["numeric_columns"]:
                st.session_state[key] = float(value)
            else:
                st.session_state[key] = str(value)


def _reset(artifact: dict) -> None:
    reference = artifact["reference_applicant"]
    for column in artifact["feature_columns"]:
        value = reference[column]
        if column in artifact["numeric_columns"]:
            st.session_state[f"field_{column}"] = float(value)
        else:
            st.session_state[f"field_{column}"] = str(value)


def _widget(column: str, artifact: dict) -> None:
    label = LABELS.get(column, column.replace("_", " "))
    help_text = HELP.get(column)
    if column in artifact["numeric_columns"]:
        bounds = artifact["numeric_bounds"][column]
        st.number_input(
            label,
            min_value=float(bounds["min"]),
            max_value=float(bounds["max"]),
            step=1.0,
            format="%.0f",
            help=help_text,
            key=f"field_{column}",
        )
        return
    st.selectbox(
        label,
        options=list(artifact["category_levels"][column]),
        help=help_text,
        key=f"field_{column}",
    )


def _collect(artifact: dict) -> dict:
    return {column: st.session_state[f"field_{column}"] for column in artifact["feature_columns"]}


def _pct(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return f"{float(value):.1%}"


def _kpi_row(artifact: dict) -> None:
    snapshot = artifact.get("test_snapshot") or {}
    cards = [
        (
            "Shipped cutoff",
            f"{float(artifact['threshold']):.2f}",
            "Deny when the score is at least this.",
        ),
        ("Holdout ROC-AUC", f"{float(snapshot['roc_auc']):.3f}", "Locked 200-applicant holdout."),
        (
            "Holdout Brier",
            f"{float(snapshot['brier']):.3f}",
            "Lower is better. Not a calibrated price.",
        ),
        (
            "Holdout cost",
            f"${float(snapshot['illustrative_cost_usd']):,.0f}",
            "Illustrative dollars on the locked holdout, not expected loss.",
        ),
    ]
    columns = st.columns(len(cards))
    for column, (label, value, help_text) in zip(columns, cards, strict=True):
        with column:
            with st.container(border=True):
                st.metric(label, value, help=help_text)


def _render_explanation(explanation) -> None:
    st.subheader("Why this score")
    st.write(explanation.lead)
    if explanation.driver_sentences:
        st.markdown("\n".join(f"- {sentence}" for sentence in explanation.driver_sentences))
        st.caption(
            "Each figure resets one field to the typical training applicant and leaves "
            "the rest as entered. Combined fields can move the score by a different "
            "amount than the sum of these lines."
        )
    if explanation.protected_note:
        st.info(explanation.protected_note)


def _render_decision(artifact: dict, record: dict) -> None:
    try:
        explanation = explain_applicant(record, artifact)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(str(exc))
        return

    result = explanation.score
    headline = "Approve — book the loan" if result.decision == "approve" else "Deny — do not book"
    if result.decision == "approve":
        st.success(headline)
    else:
        st.error(headline)
    if result.near_cutoff:
        st.warning("Within 5 points of the cutoff. Send the file to a credit officer.")

    probability, cutoff, gap = st.columns(3)
    probability.metric("Probability of default", f"{result.probability_default:.1%}")
    cutoff.metric("Desk cutoff", f"{result.threshold:.2f}")
    gap.metric("Distance to cutoff", f"{result.probability_default - result.threshold:+.1%}")
    _render_explanation(explanation)
    st.write(decision_narrative(result, artifact))


def _score_tab(artifact: dict) -> None:
    if st.button("Reset to a typical applicant"):
        _reset(artifact)
        st.rerun()

    form_column, decision_column = st.columns([1.15, 0.85], gap="large")
    with form_column:
        with st.form("applicant"):
            for title, columns, note in GROUPS:
                st.subheader(title)
                if note:
                    st.info(note)
                left, right = st.columns(2)
                for index, column in enumerate(columns):
                    with left if index % 2 == 0 else right:
                        _widget(column, artifact)
            st.form_submit_button("Score applicant", type="primary")

    record = _collect(artifact)
    with decision_column:
        st.subheader("Decision")
        _render_decision(artifact, record)


def _batch_tab(artifact: dict) -> None:
    st.write(
        "Upload a CSV with the model's columns. Extra columns, including a class label, "
        "are kept in the download. Each valid row gets a default probability, the shipped "
        "decision, and a plain-English explanation against the typical training applicant. "
        "Invalid rows stay in the file with an error and a blank score."
    )
    template = batch_template(artifact).to_csv(index=False)
    st.download_button(
        "Download a one-row template",
        data=template,
        file_name="applicant_template.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("Applications CSV", type=["csv"])
    if upload is None:
        return
    try:
        raw = pd.read_csv(upload)
        scored = score_batch(raw, artifact)
    except (ValueError, TypeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        return

    errors = scored["error"].fillna("").astype(str)
    n_bad = int((errors != "").sum())
    n_good = int(len(scored) - n_bad)
    approved = int((scored["decision"] == "approve").sum())
    denied = int((scored["decision"] == "deny").sum())
    one, two, three, four = st.columns(4)
    one.metric("Rows", f"{len(scored)}")
    two.metric("Scored", f"{n_good}")
    three.metric("Need a fix", f"{n_bad}")
    four.metric("Approved / denied", f"{approved} / {denied}")
    preview_columns = [
        column
        for column in (
            "probability_default",
            "decision",
            "near_cutoff",
            "threshold",
            "delta_vs_typical",
            "explanation",
            "error",
        )
        if column in scored.columns
    ]
    st.dataframe(scored[preview_columns], use_container_width=True, hide_index=True)
    with st.expander("Full scored file"):
        st.dataframe(scored, use_container_width=True, hide_index=True)
    st.download_button(
        "Download scored CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_applications.csv",
        mime="text/csv",
        type="primary",
    )


def _models_tab(artifact: dict) -> None:
    st.caption(
        "The shipped model is the fixed-hyperparameter candidate with the highest "
        "training-fold ROC-AUC. The tuned baseline is logistic regression with C chosen "
        "from 0.1, 1, and 10 on those same training folds. The holdout does not get a vote."
    )
    sentence = artifact.get("comparison_sentence")
    if sentence:
        st.info(sentence)
    rows = artifact.get("comparison_table") or []
    if not rows:
        st.warning("This model file has no comparison table. Retrain with `python -m credit_risk`.")
        return
    table = pd.DataFrame(rows)
    view = pd.DataFrame(
        {
            "Model": table["display_name"],
            "Role": table["role"],
            "CV ROC-AUC": table.apply(
                lambda row: f"{row['roc_auc_cv_mean']:.3f} ± {row['roc_auc_cv_std']:.3f}",
                axis=1,
            ),
            "OOF ROC-AUC": table.apply(
                lambda row: (
                    f"{row['oof_roc_auc']:.3f} "
                    f"({row['oof_roc_auc_ci_low']:.3f}–{row['oof_roc_auc_ci_high']:.3f})"
                ),
                axis=1,
            ),
            "OOF Brier": table.apply(
                lambda row: (
                    f"{row['oof_brier']:.3f} "
                    f"({row['oof_brier_ci_low']:.3f}–{row['oof_brier_ci_high']:.3f})"
                ),
                axis=1,
            ),
            "Cutoff": table["threshold"].map(lambda value: f"{float(value):.2f}"),
            "Holdout ROC-AUC": table["holdout_roc_auc"].map(lambda value: f"{float(value):.3f}"),
            "Holdout cost": table["holdout_illustrative_cost_usd"].map(
                lambda value: f"${float(value):,.0f}"
            ),
        }
    )
    st.dataframe(view, use_container_width=True, hide_index=True)
    intervals = pd.DataFrame(
        {
            "model": table["model"],
            "estimate": table["oof_roc_auc"],
            "ci_low": table["oof_roc_auc_ci_low"],
            "ci_high": table["oof_roc_auc_ci_high"],
        }
    )
    figure = draw_auc_intervals(intervals)
    st.pyplot(figure, clear_figure=True, use_container_width=True)
    st.caption(
        "Intervals are a stratified percentile bootstrap of the out-of-fold scores. "
        "The model is not refit inside the bootstrap."
    )


def _policy_tab(artifact: dict) -> None:
    desk = artifact.get("desk")
    if not desk:
        st.warning("This model file has no score curves. Retrain with `python -m credit_risk`.")
        return
    ratio = st.slider(
        "Cost of a false approval, relative to a false denial",
        min_value=1,
        max_value=20,
        value=int(artifact.get("cost_false_approval", 5)),
        step=1,
        help=(
            "The published Statlog matrix uses 5. One false denial stays 1 cost unit, "
            "drawn as $200. Dollars are illustrative."
        ),
    )
    source = st.radio(
        "Scores behind the curve",
        ["Training out-of-fold", "Locked holdout"],
        horizontal=True,
        help="The shipped cutoff was chosen on the training out-of-fold scores.",
    )
    if source == "Locked holdout":
        labels = np.asarray(desk["holdout_y"], dtype=int)
        scores = np.asarray(desk["holdout_scores"], dtype=float)
        brier = float(desk["holdout_brier"])
        constant = float(desk["holdout_brier_constant"])
        curve_title = "Locked holdout calibration"
    else:
        labels = np.asarray(desk["oof_y"], dtype=int)
        scores = np.asarray(desk["oof_scores"], dtype=float)
        brier = float(desk["oof_brier"])
        constant = float(desk["oof_brier_constant"])
        curve_title = "Out-of-fold calibration"
    sweep = policy_curve(labels, scores, float(ratio), 1.0)
    chosen = choose_threshold(sweep)
    picked = row_at_threshold(sweep, chosen)
    shipped_cutoff = round(float(artifact["threshold"]), 2)
    try:
        shipped = row_at_threshold(sweep, shipped_cutoff)
    except ValueError:
        shipped = None

    cost, savings, booked, rate = st.columns(4)
    cost.metric(
        "Illustrative cost at the best cutoff", f"${float(picked['illustrative_cost_usd']):,.0f}"
    )
    savings.metric(
        "Profit vs approving everyone",
        f"${float(picked['savings_vs_approve_all_usd']):,.0f}",
    )
    booked.metric("Best cutoff", f"{chosen:.2f}")
    rate.metric("Approval rate", f"{float(picked['approval_rate']):.0%}")
    if shipped is not None:
        st.caption(
            f"At this loss ratio the shipped cutoff {shipped_cutoff:.2f} costs "
            f"${float(shipped['illustrative_cost_usd']):,.0f} "
            f"({float(shipped['savings_vs_approve_all_usd']):,.0f} versus approving everyone). "
            "Moving the slider does not change the model saved in this app."
        )
    profit = draw_profit(sweep, chosen)
    st.pyplot(profit, clear_figure=True, use_container_width=True)
    st.caption(
        "Profit is illustrative dollars saved versus approving every applicant in the "
        "selected scores, at the loss ratio on the slider. It is not revenue and not expected loss."
    )
    reliability = draw_calibration(labels, scores, title=curve_title)
    st.pyplot(reliability, clear_figure=True, use_container_width=True)
    st.caption(
        f"Brier score {brier:.3f}. Predicting this split's own default rate for every "
        f"applicant scores {constant:.3f}. Balanced class weights mean the raw score is "
        "a ranking device. Read the curve before treating it as a probability."
    )


def _rate_table(records: list[dict]) -> None:
    table = pd.DataFrame(records)
    if table.empty:
        st.write("No group rows were stored.")
        return
    view = pd.DataFrame(
        {
            "Split of the file": table["family"],
            "Group": table["group"],
            "Applicants": table["n"].astype(int),
            "Defaulters": table["n_default"].astype(int),
            "Approval rate": table["approval_rate"].map(_pct),
            "False approval rate": table["false_approval_rate"].map(_pct),
            "False denial rate": table["false_denial_rate"].map(_pct),
        }
    )
    st.dataframe(view, use_container_width=True, hide_index=True)
    chart = table.copy()
    chart["label"] = chart["family"].astype(str) + " · " + chart["group"].astype(str)
    st.bar_chart(chart.set_index("label")["approval_rate"], height=260)


def _fairness_tab(artifact: dict) -> None:
    report = artifact.get("group_fairness")
    if not report:
        st.warning("This model file has no fairness table. Retrain with `python -m credit_risk`.")
        return
    st.write(report["oof_note"])
    st.subheader("Training out-of-fold scores")
    _rate_table(report["oof_groups"])
    st.caption(
        "False approval rate is the share of defaulters who were booked. "
        "False denial rate is the share of people who paid and were turned away."
    )
    with st.expander("Locked holdout, same shipped cutoff"):
        st.write(report["holdout_note"])
        _rate_table(report["holdout_groups"])


def main() -> None:
    st.set_page_config(
        page_title="Credit desk",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    try:
        artifact = cached_artifact()
    except (FileNotFoundError, ValueError) as exc:
        st.title("Credit desk")
        st.error(str(exc))
        st.stop()

    _seed(artifact)
    st.title("Credit desk")
    st.caption(
        f"{artifact.get('display_name', 'Saved model')}. "
        "Statlog German credit. The cutoff minimizes the published cost of a false "
        "approval versus a false denial. Startup reads the saved model from disk and "
        "does not download data."
    )
    _kpi_row(artifact)

    with st.sidebar:
        st.header("Shipped policy")
        st.metric("Cutoff", f"{float(artifact['threshold']):.2f}")
        st.write(f"Model: {DISPLAY_NAMES.get(artifact['model_name'], artifact['model_name'])}")
        st.write(
            f"False approval costs {float(artifact['cost_false_approval']):g} "
            f"(${float(artifact['illustrative_false_approval_usd']):,.0f} illustrative). "
            f"False denial costs {float(artifact['cost_false_denial']):g} "
            f"(${float(artifact['illustrative_denial_usd']):,.0f} illustrative)."
        )
        st.caption("One cost unit is shown as $200. That is a translation, not a loss estimate.")

    score_tab, batch_tab, models_tab, policy_tab, fairness_tab = st.tabs(
        ["Score", "Batch", "Models", "Policy", "Fairness"]
    )
    with score_tab:
        _score_tab(artifact)
    with batch_tab:
        _batch_tab(artifact)
    with models_tab:
        _models_tab(artifact)
    with policy_tab:
        _policy_tab(artifact)
    with fairness_tab:
        _fairness_tab(artifact)


if __name__ == "__main__":
    main()
