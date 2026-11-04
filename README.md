# Credit risk decisions, not just a leaderboard

**Ladarius Jackson · Business Analytics**

[![CI](https://github.com/ladariusjackson1/credit-risk-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ladariusjackson1/credit-risk-prediction/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/Live_demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://credit-risk-prediction.streamlit.app)

## Business problem

A lender does not get paid for a high AUC. It gets paid when a booked loan performs, and it takes a loss when a booked loan defaults. This project asks a desk-level question of the public Statlog German Credit file: **given that funding a defaulter is five times as costly as turning away a borrower who would have paid, who should be approved?**

The answer is a gradient-boosting score with a cutoff of **0.15**, not the usual 0.50. On a locked holdout of 200 applicants the policy books 56 loans, 5 of which default. The illustrative cost of that book is **$22,800**.

## Money saved

Dollars below translate the published Statlog matrix at **1 cost unit = $200**. A false approval is $1,000. A false denial is $200 of margin given up. They are not a loss model fit from the file.

| Holdout policy | Booked defaults | Illustrative cost | Saved by the 0.15 cutoff |
| --- | ---: | ---: | ---: |
| Approve everyone | 60 | $60,000 | $37,200 |
| Deny everyone | 0 | $28,000 | $5,200 |
| Score cutoff 0.50 | 20 | $27,200 | $4,400 |
| **Tuned cutoff 0.15** | **5** | **$22,800** | — |

Against approving every holdout applicant, the tuned desk avoids 55 false approvals and gives up 89 performing loans. That is $55,000 of default losses avoided and $17,800 of margin given up, a net **$37,200**. Against a 0.50 cutoff, booked defaults fall from 20 to 5. The $15,000 of default losses avoided outweighs $10,600 of extra turned-away margin, a net **$4,400**.

## Desk demo

![Desk scorer: a typical applicant, then the fields that push the default probability up or down](docs/credit-desk-demo.gif)

The [live demo](https://credit-risk-prediction.streamlit.app) runs this repository on Streamlit Community Cloud. The main file is `streamlit_app.py`. The app loads `models/credit_risk_model.joblib` from disk and does not download the dataset at startup. For anyone other than the typical training applicant, the panel says which fields push the estimated default probability up or down relative to that profile.

## The decision

The positive class is a default. In the file that label is `bad` (300 of 1,000 applicants). A **false approval** is a default that was booked. A **false denial** is a performing borrower who was turned away.

Statlog publishes the cost matrix used here: a false approval costs 5, a false denial costs 1 ([UCI documentation](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)). Dollars in this write-up are a translation of that matrix, not a loss model fit from the file: **1 cost unit = $200**, so a false denial is $200 of margin given up and a false approval is $1,000. Change the ratio and the cutoff moves. That sensitivity is part of the result.

## Data

[Hofmann, H. (1994). Statlog (German Credit Data)](https://doi.org/10.24432/C5NC77), read as OpenML `credit-g` version 1. One thousand installment applicants, 20 attributes, no missing values, a 30% default rate. `python -m credit_risk` downloads the file and writes `data/german_credit.csv`. If OpenML is unreachable, the same command uses that snapshot. If both are missing, it tells you where to put the file.

`personal_status` encodes sex and marital status. `foreign_worker` encodes nationality. They stay in the benchmark model because they are columns in the public file. A second cross-validation drops them. They should not be inputs to a live U.S. decision.

## Design

Nothing in the model choice or the cutoff sees the holdout.

1. Stratified 80/20 split (`random_state=42`): 800 train, 200 test.
2. On the training rows only, three pipelines are compared with 5-fold stratified cross-validation. Each pipeline imputes, scales numerics, and one-hot encodes categories, then fits logistic regression, a random forest, or histogram gradient boosting. All three use balanced class weights so the 30/70 base rate does not dominate fitting.
3. The selected model is the one with the highest mean cross-validated ROC-AUC. Ties would break toward a tighter fold spread, then the model name. Hyperparameters were fixed before the run. This is a comparison of model classes, not a search that was allowed to wander.
4. Each model gets its own cutoff by minimizing Statlog cost on **out-of-fold training scores**. Equal costs keep the higher cutoff, which books more loans once the cost is the same.
5. The chosen pipelines are refit on the 800 and scored **once** on the 200. Permutation importance is computed there, on the original columns, so the chart says `checking_status` rather than a one-hot fragment. No SHAP.

Class weights and the cutoff do different jobs. Weights change how the model is fit. The cutoff is where the desk says yes. Because the weights are balanced, a score of 0.15 is **not** a claim that the applicant has a 15% chance of default. The holdout calibration plot is the check on that.

## Results

Cross-validated ROC-AUC on the 800, then the locked holdout. Each model is shown at the cutoff chosen from its own out-of-fold scores.

| Model | CV ROC-AUC | Holdout ROC-AUC | Holdout average precision | Cutoff | Approved | Defaults denied | Default rate among booked | Illustrative holdout cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.769 ± 0.059 | 0.806 | 0.633 | 0.28 | 36% | 88% | 9.7% | $22,000 |
| Random forest | 0.792 ± 0.047 | 0.805 | 0.644 | 0.36 | 37% | 90% | 8.1% | $20,400 |
| **Gradient boosting** | **0.793 ± 0.054** | **0.772** | **0.600** | **0.15** | **28%** | **92%** | **8.9%** | **$22,800** |
| Approve everyone | — | — | — | — | 100% | 0% | 30% | $60,000 |
| Deny everyone | — | — | — | — | 0% | 100% | — | $28,000 |

Gradient boosting is the selected model because its training-fold ROC-AUC is the highest. The margin over the random forest is 0.0005, against a fold standard deviation near 0.05, so the two tree models are not separable on this sample. Out-of-fold cost ranks them the same way: boosting 405, logistic regression 410, forest 420, on the 800. The holdout then flips the cost ranking (forest $20,400, logistic $22,000, boosting $22,800). That flip is reported and **not** used to change the model. Two hundred rows can move a couple of defaults. The precommitted rule already picked the out-of-fold cost winner, and every tuned policy beats both naive baselines.

![Training-fold ROC-AUC and average precision](reports/figures/model_comparison.png)

![Holdout ROC curves](reports/figures/roc_curves.png)

![Holdout precision-recall curves](reports/figures/precision_recall.png)

### The cutoff is the product

On the training out-of-fold scores, cost bottoms at 0.15. Denying everyone costs 560. Approving everyone costs 1,200. The 0.50 score cutoff is not on that minimum.

![Cost against the cutoff](reports/figures/threshold_cost.png)

On the holdout, the same 0.15 rule versus a 0.50 rule:

| Holdout decision | Approved and paid | Denied but would have paid | Approved and defaulted | Denied and would have defaulted | Cost | Illustrative $ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cutoff 0.50 | 104 | 36 | 20 | 40 | 136 | $27,200 |
| **Cutoff 0.15** | **51** | **89** | **5** | **55** | **114** | **$22,800** |

The 0.50 rule books 20 defaulters. At $1,000 each that is already $20,000, plus $7,200 of turned-away margin, $27,200 total — $800 under the cost of lending to nobody. The cost-minimizing rule books 5 defaulters and turns away more good applications. That trade is what the 5:1 matrix asks for. The booked book’s default rate is 8.9% (5 of 56) against a 30% base rate.

![Holdout confusion matrices](reports/figures/confusion_matrices.png)

![Illustrative holdout cost of each policy](reports/figures/policy_cost.png)

![Holdout scores by outcome](reports/figures/score_distribution.png)

### If management changes the loss ratio

The cutoff below is retuned on the same out-of-fold boosting scores. Only the costs change.

| False approval : false denial | Illustrative loss on a default | Cutoff | Share approved | Share of defaulters denied |
| --- | ---: | ---: | ---: | ---: |
| 2 : 1 | $400 | 0.47 | 62% | 73% |
| **5 : 1 (published)** | **$1,000** | **0.15** | **31%** | **93%** |
| 10 : 1 | $2,000 | 0.15 | 31% | 93% |

Moving from 5:1 to 2:1 opens the book from about 31% approved to 62%. Moving from 5:1 to 10:1 does not move the cutoff: on these scores, 0.15 is already the conservative pocket, and the next lower cutoffs deny more good borrowers than the extra defaults they catch are worth. The desk should retune when the loss ratio changes, not when a new model is fashionable.

### What the score is actually using

Permutation importance shuffles one original column at a time and records the drop in holdout ROC-AUC. Checking-account status is the only column that moves AUC by a wide margin (about 0.08, with a repeat-to-repeat spread of 0.02). Purpose, credit amount, credit history, and duration form a second tier around 0.02. Several columns, including age, sit on zero.

![Permutation importance](reports/figures/permutation_importance.png)

The raw rates explain the checking-account result, and they also warn against reading the codebook as modern American English. In this file, **no checking account is the low-risk group** (12% default, 46 of 394). A negative checking balance is the high-risk group (49%, 135 of 274). Balances of 0 to 200 default at 39%; balances of 200 or more default at 22%. The book rate is 30%.

Credit history has the same kind of trap. "Critical / other existing credit" defaults at 17% (50 of 293). "No credits / all paid" defaults at 62% (25 of 40). The model can use the column because it is allowed to learn those rates. A reviewer who relabels them by the English words will make the model worse.

Loan term and size show up in the same direction a credit officer expects, with a bend in the middle of the amount distribution. Terms of 12 months or less default at 21% (76 of 359). Terms above 24 months default at 44% (102 of 230). The largest amount quartile, above about 3,972 DM, defaults at 42%. The quartile below it defaults at 22%. Amount is not a straight line.

![Default rate by checking status and credit amount](reports/figures/segment_default_rates.png)

### Protected attributes

Dropping `personal_status` and `foreign_worker` and repeating the same 5 folds changes boosting's mean ROC-AUC from 0.793 to 0.786. That is a loss of about 0.007. On the holdout permutation test, shuffling `personal_status` does not hurt AUC (the mean drop is negative), and `foreign_worker` is inside its own noise. Ranking quality does not depend on sex, marital status, or nationality here. A policy build should refit without those two columns. The saved scorer still asks for them so that a score matches the benchmark model that was trained.

### Calibration

Holdout Brier score for the selected model is 0.187. Predicting the 30% base rate for everyone would score 0.21, so the probabilities are somewhat informative and still noisy on 200 rows. Use them to rank and to set a cutoff. Do not book a loss reserve off the raw score.

![Holdout calibration](reports/figures/calibration.png)

## Findings

- Under the published 5:1 matrix, the value is in the cutoff. A 0.50 threshold almost fails to beat "lend to nobody." The tuned rule cuts booked defaults from 20 to 5 on the holdout and lowers illustrative cost from $27,200 to $22,800.
- The tuned book is small on purpose: 28% of holdout applicants, an 8.9% default rate inside the booked loans, and 92% of defaulters turned away.
- Boosting, the forest, and logistic regression are close. The written rule selects boosting. The holdout cost ranking prefers the forest by $2,400, which is not a reason to reopen the choice after seeing the test set.
- Checking-account status dominates. Read the empirical rates, not the English gloss. "No checking account" is 12% default in this file.
- Sex, marital status, and nationality can come out. The cross-validated AUC cost is about 0.007.
- If a false approval is only twice a false denial, the same scores support approving about 62% instead of 31%. The cutoff is a management input.

## What this should not be used for

This is a 1994 German installment book with 1,000 rows. It is not a 2026 U.S. application population, it is not a pricing model, and it is not fair-lending validation. The dollar figures exist so a reader can feel the cost matrix. They are not expected loss. Scores were not calibrated to a live portfolio's base rate.

## How to run

Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m credit_risk
PYTHONPATH=src python -m pytest
python -m ruff check src tests streamlit_app.py
python -m streamlit run streamlit_app.py
```

`requirements.txt` pins every direct dependency, including scikit-learn 1.9.1, the version that wrote `models/credit_risk_model.joblib`. `python -m credit_risk` rewrites `reports/`, the figures, and the model. `--refresh` redownloads OpenML `credit-g` before training. Without `--refresh`, training reads `data/german_credit.csv`.

Streamlit Community Cloud deploys this repository with main file `streamlit_app.py` and Python 3.12 (`.python-version`). `.streamlit/config.toml` turns usage stats off and sets the desk theme. The app adds `src` to its import path, loads the committed model, and does not contact the network on startup. It opens on a typical training applicant. Reset returns to that profile. When a field differs from that profile, the decision panel states the local contribution: the change in estimated default probability from putting that one field back, with the other fields left as entered.

The numbers in this README were produced by that command on Python 3.12 with scikit-learn 1.9.1, pandas 3.1.0, and NumPy 2.5.4 (`reports/metrics.json` records the versions).

## Layout

```
streamlit_app.py                 Community Cloud entry point
src/credit_risk/                 data, features, models, cost, plots, scoring, explanations
data/german_credit.csv           offline snapshot of OpenML credit-g
models/credit_risk_model.joblib
reports/                         metrics, tables, figures
docs/credit-desk-demo.gif
tests/                           pytest, 100% line coverage of src/credit_risk
.streamlit/config.toml
.github/workflows/ci.yml
```

Continuous integration on GitHub Actions runs Ruff and pytest. The suite covers the OpenML fallback, the cost arithmetic, cutoff ties, each model, the end-to-end study on a synthetic book, and the saved-model scorer. It does not need a network call.

## Reproduce the tables

| File | What it is |
| --- | --- |
| `reports/metrics.json` | The study record, including fold means and the holdout |
| `reports/model_comparison.csv` | Cross-validated ROC-AUC and average precision |
| `reports/test_metrics.csv` | Holdout operating point for each model |
| `reports/threshold_sweep.csv` | Cost at every cutoff |
| `reports/cost_sensitivity.csv` | Cutoff at loss ratios 2:1, 5:1, and 10:1 |
| `reports/permutation_importance.csv` | Holdout permutation importance |
| `reports/categorical_default_rates.csv` | Default rate by category level |
| `reports/numeric_band_default_rates.csv` | Default rate by quantile band |
