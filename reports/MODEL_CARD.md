# Model card: German credit desk score

## Model details

This card describes the scorer written by `python -m credit_risk` and saved at `models/credit_risk_model.joblib`.

- **Shipped model:** Gradient boosting (`gradient_boosting`).
- **Decision:** approve when the estimated probability of default is below 0.15; otherwise deny.
- **Positive class:** default (dataset label 'bad').
- **Training rows:** 800. **Locked holdout:** 200. Stratified split, `random_state=42`, test size 20%.
- **Cross-validation:** 5 stratified folds on the training rows.
- **Selection rule:** Highest mean stratified cross-validated ROC-AUC on the training set. Ties break toward the smaller fold standard deviation, then the model name.
- **Cutoff rule:** On training out-of-fold scores, choose the cutoff with the lowest Statlog expected cost. Equal costs keep the higher cutoff.
- **Environment:** Python 3.12.3, scikit-learn 1.9.1, pandas 3.1.0, NumPy 2.5.4.

The shipped model is the winner of a fixed-hyperparameter race among logistic regression, a random forest, and histogram gradient boosting. A tuned logistic baseline is reported beside it and is not allowed to replace the shipped model after the holdout is opened.

## Intended use

The score is a portfolio demonstration of a cost-sensitive credit decision on a public 1994 German installment file. It is meant to rank applicants from that file and to show how a published loss ratio moves the approval cutoff.

A suitable use is a credit-risk, analytics, or model-risk discussion: what the cutoff costs, how uncertain the model ranking is, and where approval rates differ by age and by the file's sex label.

## Out of scope

- A live lending decision, a pricing engine, a credit limit, or an adverse-action notice.
- Expected loss, capital, reserves, or a dollar forecast. The dollars in this project translate the Statlog cost matrix at $200 per cost unit. A false approval is 5 units ($1,000). A false denial is 1 unit ($200).
- Any population other than this file. It is not a 2026 book and it is not a U.S. application population.
- Fair-lending compliance, disparate-impact testing, or a decision about a protected class.

## Data

- **Source:** Statlog German Credit (OpenML credit-g, version 1).
- **Citation:** Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.
- **Rows:** 1000. **Default rate:** 30.0% (the dataset label `bad`).
- **Missing numeric values:** 0. **Missing categorical values:** 0.
- **Inputs:** the 20 Statlog attributes. Numeric fields are median-imputed and scaled. Categorical fields are mode-imputed and one-hot encoded. Unknown categories at score time become zeros.
- **Sensitive inputs kept in the benchmark:** `personal_status` (sex and marital status together) and `foreign_worker` (nationality). They are in the public file, so the benchmark can be reproduced. They should not be inputs to a live U.S. decision.

## Metrics

Bootstrap method: Stratified percentile bootstrap of fixed scores. Each replicate draws applicants with replacement inside the default class and inside the paid class, keeping those class counts. The model is not refit. The interval does not include the noise of retraining or, for the tuned logistic baseline, the noise of choosing C again. Replicates: 1000. Interval: 95% percentile. Seed: 42.

Tuned baseline: Tuned logistic baseline, logistic regression with C = 0.1 chosen from {0.1, 1, 10}. On the training set only, C is chosen from the fixed grid 0.1, 1, and 10 by mean stratified cross-validated ROC-AUC. Ties prefer the C closest to 1, then the smaller C. C is chosen on the same folds that are reported, so the baseline's cross-validated AUC can look slightly better than a nested estimate. The holdout is not used to choose C.

On out-of-fold scores, Gradient boosting minus Tuned logistic baseline has a ROC-AUC difference of +0.010 (95% bootstrap interval -0.010 to 0.031). The interval includes zero, so this training sample does not separate them on ROC-AUC.

The out-of-fold Brier difference, shipped model minus baseline, is -0.011 (95% interval -0.020 to -0.002). A negative Brier difference means the shipped scores are closer to the observed outcomes. The out-of-fold cost-per-applicant difference at each model's own cutoff is 0.001 (95% interval -0.054 to 0.056). A negative cost difference means the shipped policy costs less on these out-of-fold scores. Holdout illustrative cost is $22,800 for the shipped model and $21,000 for the tuned baseline. The holdout was not used to pick the model or the cutoff.

| Model | Role | CV ROC-AUC | OOF ROC-AUC (95% CI) | OOF Brier (95% CI) | Cutoff | Holdout ROC-AUC | Holdout Brier | Holdout illustrative cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | Candidate | 0.769 ± 0.059 | 0.767 (0.732 to 0.801) | 0.196 (0.180 to 0.212) | 0.28 | 0.806 | 0.182 | $22,000 |
| Random forest | Candidate | 0.792 ± 0.047 | 0.791 (0.757 to 0.823) | 0.180 (0.172 to 0.190) | 0.36 | 0.805 | 0.180 | $20,400 |
| Gradient boosting | Selected model | 0.793 ± 0.054 | 0.790 (0.756 to 0.822) | 0.178 (0.163 to 0.195) | 0.15 | 0.772 | 0.187 | $22,800 |
| Tuned logistic baseline | Tuned baseline | 0.781 ± 0.057 | 0.780 (0.746 to 0.813) | 0.189 (0.176 to 0.202) | 0.34 | 0.809 | 0.182 | $21,000 |

CV ROC-AUC is the mean across training folds, with the fold standard deviation. OOF metrics are computed on the stacked out-of-fold scores. The interval is the bootstrap described above. Holdout columns are a single locked look.

### Calibration

Class weights are balanced, so a score of 0.15 is not a claim that the applicant has a 15% chance of default. On the holdout, the shipped model's Brier score is 0.187. Predicting the holdout's own default rate for every applicant scores 0.210. On the training out-of-fold scores the Brier score is 0.178, against 0.210 for the constant out-of-fold default rate. Use the score to rank and to set a cutoff. Do not book a loss reserve from the raw score. The reliability diagram is `reports/figures/calibration.png`.

## Fairness

Sex in this check is the first word of `personal_status`. personal_status values that start with 'female' or 'male'; marital status inside the label is collapsed. Age bands are split at 25.

### Training out-of-fold scores

At cutoff 0.15 on the training out-of-fold scores (800 applicants), a file is approved when its score is below the cutoff.

On this split, female-labeled applicants were approved 24.9% of the time (n=249) and male-labeled applicants were approved 34.3% of the time (n=551). The approval-rate gap, female-labeled applicants minus male-labeled applicants, is -9.4 percentage points. Among people who defaulted, the share who were still approved was 6.7% for female-labeled applicants (89 defaulters) and 6.6% for male-labeled applicants (151 defaulters). That gap is +0.1 percentage points. Among people who paid, the share who were turned away was 65.0% for female-labeled applicants (160 who paid) and 55.2% for male-labeled applicants (400 who paid). That gap is +9.8 percentage points. The approval ratio of female-labeled applicants to male-labeled applicants is 0.73. A ratio of 0.80 is a common U.S. screening rule of thumb, and it is not a legal result on this file.

On this split, applicants under 25 were approved 13.6% of the time (n=110) and applicants 25 and older were approved 34.2% of the time (n=690). The approval-rate gap, applicants under 25 minus applicants 25 and older, is -20.6 percentage points. Among people who defaulted, the share who were still approved was 4.8% for applicants under 25 (42 defaulters) and 7.1% for applicants 25 and older (198 defaulters). That gap is -2.3 percentage points. Among people who paid, the share who were turned away was 80.9% for applicants under 25 (68 who paid) and 54.9% for applicants 25 and older (492 who paid). That gap is +26.0 percentage points. The approval ratio of applicants under 25 to applicants 25 and older is 0.40. A ratio of 0.80 is a common U.S. screening rule of thumb, and it is not a legal result on this file.

Read these gaps as a description of this cutoff on this split, not as a finding of discrimination. personal_status in the 1994 file mixes sex and marital status, so the female and male rows collapse those labels. The age split is under 25 versus 25 and older, which is a reporting choice. Age is an input to the model, and personal status is an input to the benchmark model. A cell with few defaults moves when one or two people change outcome. This is not a fair-lending review, not a U.S. applicant population, and not a reason by itself to approve or deny anyone.

### Locked holdout

At cutoff 0.15 on the locked holdout (200 applicants), a file is approved when its score is below the cutoff.

On this split, female-labeled applicants were approved 26.2% of the time (n=61) and male-labeled applicants were approved 28.8% of the time (n=139). The approval-rate gap, female-labeled applicants minus male-labeled applicants, is -2.5 percentage points. Among people who defaulted, the share who were still approved was 10.0% for female-labeled applicants (20 defaulters) and 7.5% for male-labeled applicants (40 defaulters). That gap is +2.5 percentage points. Among people who paid, the share who were turned away was 65.9% for female-labeled applicants (41 who paid) and 62.6% for male-labeled applicants (99 who paid). That gap is +3.2 percentage points. The approval ratio of female-labeled applicants to male-labeled applicants is 0.91. A ratio of 0.80 is a common U.S. screening rule of thumb, and it is not a legal result on this file.

On this split, applicants under 25 were approved 15.4% of the time (n=39) and applicants 25 and older were approved 31.1% of the time (n=161). The approval-rate gap, applicants under 25 minus applicants 25 and older, is -15.7 percentage points. Among people who defaulted, the share who were still approved was 10.5% for applicants under 25 (19 defaulters) and 7.3% for applicants 25 and older (41 defaulters). That gap is +3.2 percentage points. Among people who paid, the share who were turned away was 80.0% for applicants under 25 (20 who paid) and 60.8% for applicants 25 and older (120 who paid). That gap is +19.2 percentage points. The approval ratio of applicants under 25 to applicants 25 and older is 0.50. A ratio of 0.80 is a common U.S. screening rule of thumb, and it is not a legal result on this file.

Read these gaps as a description of this cutoff on this split, not as a finding of discrimination. personal_status in the 1994 file mixes sex and marital status, so the female and male rows collapse those labels. The age split is under 25 versus 25 and older, which is a reporting choice. Age is an input to the model, and personal status is an input to the benchmark model. A cell with few defaults moves when one or two people change outcome. This is not a fair-lending review, not a U.S. applicant population, and not a reason by itself to approve or deny anyone.

Dropping `personal_status` and `foreign_worker` and repeating the same folds changes the shipped model's mean ROC-AUC from 0.793 to 0.786 (change when dropped: -0.007). Ranking quality on this file does not depend on those two columns in any way that this sample can pin down. A policy build should refit without them. The saved scorer still asks for them so a score matches the benchmark that was trained.

## Factors

The largest holdout permutation drop is `checking_status` at 0.083 ROC-AUC (repeat spread 0.021). Permutation importance shuffles one original column at a time on the holdout. The codebook is not modern American English: in this file, having no checking account is the low-risk checking status. Read the empirical rates in `reports/categorical_default_rates.csv` before reading the English gloss.

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
