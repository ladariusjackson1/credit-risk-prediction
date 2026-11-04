"""Desk tool: score one German-credit applicant with the saved model.

Run from the repository root:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from credit_risk.config import DISPLAY_NAMES
from credit_risk.scoring import decision_narrative, load_artifact, score_applicant

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


def _render_decision(artifact: dict, record: dict) -> None:
    try:
        result = score_applicant(record, artifact)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(str(exc))
        return

    headline = "Approve — book the loan" if result.decision == "approve" else "Deny — do not book"
    if result.decision == "approve":
        st.success(headline)
    else:
        st.error(headline)
    if result.near_cutoff:
        st.warning("Within 5 points of the cutoff. Send the file to a credit officer.")

    probability, cutoff, gap = st.columns(3)
    probability.metric("Probability of default", f"{result.probability_default:.1%}")
    cutoff.metric("Desk cutoff", f"{result.threshold:.1%}")
    gap.metric("Distance to cutoff", f"{result.probability_default - result.threshold:+.1%}")
    st.write(decision_narrative(result, artifact))

    reference_score = score_applicant(artifact["reference_applicant"], artifact)
    delta = result.probability_default - reference_score.probability_default
    st.caption(
        "A typical training applicant scores "
        f"{reference_score.probability_default:.1%}. "
        f"This file is {delta:+.1%} relative to that profile."
    )

    snapshot = artifact.get("test_snapshot", {})
    if snapshot:
        st.caption(
            f"Locked holdout for {artifact['display_name']}: "
            f"ROC-AUC {snapshot['roc_auc']:.3f}, "
            f"approval rate {snapshot['approval_rate']:.0%}, "
            f"share of defaulters denied {snapshot['recall_bad']:.0%}."
        )


def main() -> None:
    st.set_page_config(page_title="Credit desk scorer", layout="wide")
    try:
        artifact = load_artifact()
    except (FileNotFoundError, ValueError) as exc:
        st.title("Credit desk scorer")
        st.error(str(exc))
        st.stop()

    _seed(artifact)
    st.title("Credit desk scorer")
    st.caption(
        "Scores one installment applicant with the model selected in this study. "
        "The cutoff minimizes the Statlog cost of a false approval versus a false denial."
    )

    with st.sidebar:
        st.header("Policy")
        st.metric("Cutoff", f"{artifact['threshold']:.0%}")
        st.write(f"Model: {DISPLAY_NAMES.get(artifact['model_name'], artifact['model_name'])}")
        st.write(
            f"False approval costs {artifact['cost_false_approval']:g} "
            f"(${artifact['illustrative_false_approval_usd']:,.0f} illustrative). "
            f"False denial costs {artifact['cost_false_denial']:g} "
            f"(${artifact['illustrative_denial_usd']:,.0f} illustrative)."
        )
        st.caption(
            "Dollars are a translation of the published 5:1 matrix, not a loss estimate "
            "from this file. One cost unit is shown as $200."
        )
        top = pd.DataFrame(artifact.get("top_features", []))
        if not top.empty:
            st.subheader("What the holdout says matters")
            chart = top.set_index("feature")["importance_mean"].sort_values()
            st.bar_chart(chart)
            st.caption("Drop in holdout ROC-AUC when that column is shuffled.")

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


if __name__ == "__main__":
    main()
