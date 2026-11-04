"""Credit-risk study on the Statlog German Credit dataset.

The positive class is a default (the dataset label ``bad``). Decisions are
made by comparing a model's score with a cutoff that minimizes the Statlog
cost of false approvals and false denials.
"""

from credit_risk.scoring import load_artifact, score_applicant

__version__ = "1.0.0"

__all__ = ["__version__", "load_artifact", "score_applicant"]
