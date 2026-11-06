"""Paths, schema, and the credit policy this study optimizes."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CACHE_PATH = DATA_DIR / "german_credit.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "credit_risk_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_IMPORTANCE_REPEATS = 20
N_BOOTSTRAP = 1000
BOOTSTRAP_ALPHA = 0.05
# Young-applicant cut used only for the fairness table. It is a reporting
# choice, not a class the model was trained to recognize.
AGE_GROUP_CUT = 25.0
MAX_BATCH_ROWS = 5000

OPENML_NAME = "credit-g"
OPENML_VERSION = 1
DATASET_NAME = "Statlog German Credit (OpenML credit-g, version 1)"
DATASET_CITATION = (
    "Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77"
)

TARGET_COLUMN = "class"
POSITIVE_LABEL = "bad"
NEGATIVE_LABEL = "good"

# Published Statlog matrix: rows are actual class, columns are the decision.
# Funding a defaulter (actual bad, predicted good) costs 5. Turning away a
# borrower who would have paid (actual good, predicted bad) costs 1.
COST_FALSE_APPROVAL = 5.0
COST_FALSE_DENIAL = 1.0
COST_RATIOS = ((2.0, 1.0), (5.0, 1.0), (10.0, 1.0))

# Illustration only: one Statlog cost unit is treated as $200 of contribution
# margin, so the published 5:1 matrix is $1,000 lost on a false approval and
# $200 of margin given up on a false denial. These are not estimated from the file.
ILLUSTRATIVE_DENIAL_USD = 200.0

# Scores within this distance of the cutoff are flagged for a credit officer.
REVIEW_BAND = 0.05

NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

# Sex/marital status and nationality. Kept in the benchmark model because they
# are columns in the public file, then removed in a side-by-side comparison.
PROTECTED_FEATURES = ["personal_status", "foreign_worker"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Column order of the OpenML / UCI file. The saved snapshot uses this order.
STUDY_COLUMNS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    TARGET_COLUMN,
]

MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")

# Logistic regression is the simple baseline. C is chosen on the training
# folds from this grid, which was fixed before any holdout look. The grid is
# not part of the model-class race: the shipped model is still the fixed
# candidate with the highest training-fold ROC-AUC.
BASELINE_FAMILY = "logistic_regression"
TUNED_BASELINE_NAME = "tuned_logistic"
LOGISTIC_C_GRID = (0.1, 1.0, 10.0)

DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "tuned_logistic": "Tuned logistic baseline",
}

# Fixed before looking at the holdout. This is a model-class comparison,
# not a search that was allowed to peek at the test set.
LOGISTIC_MAX_ITER = 1000
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 8
RF_MIN_SAMPLES_LEAF = 5
GB_MAX_ITER = 200
GB_LEARNING_RATE = 0.08
GB_MAX_DEPTH = 3
GB_MIN_SAMPLES_LEAF = 20

CATEGORY_ORDERS = {
    "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
    "credit_history": [
        "no credits/all paid",
        "all paid",
        "existing paid",
        "delayed previously",
        "critical/other existing credit",
    ],
    "purpose": [
        "new car",
        "used car",
        "furniture/equipment",
        "radio/tv",
        "domestic appliance",
        "repairs",
        "education",
        "retraining",
        "business",
        "other",
    ],
    "savings_status": ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
    "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    "personal_status": [
        "female div/dep/mar",
        "male single",
        "male mar/wid",
        "male div/sep",
    ],
    "other_parties": ["none", "co applicant", "guarantor"],
    "property_magnitude": ["real estate", "life insurance", "car", "no known property"],
    "other_payment_plans": ["bank", "stores", "none"],
    "housing": ["rent", "own", "for free"],
    "job": [
        "unemp/unskilled non res",
        "unskilled resident",
        "skilled",
        "high qualif/self emp/mgmt",
    ],
    "own_telephone": ["none", "yes"],
    "foreign_worker": ["yes", "no"],
}

ARTIFACT_VERSION = 2
