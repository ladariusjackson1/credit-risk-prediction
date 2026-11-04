"""Small labeled frames that match the German-credit schema. Used only by tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_credit_frame(n: int = 160, seed: int = 7) -> pd.DataFrame:
    """Build a schema-valid book with a mild default signal. Not the real dataset."""
    rng = np.random.default_rng(seed)
    checking = rng.choice(
        ["<0", "0<=X<200", ">=200", "no checking"],
        size=n,
        p=[0.28, 0.27, 0.07, 0.38],
    )
    duration = rng.integers(6, 49, size=n)
    amount = rng.integers(400, 9000, size=n)
    history = rng.choice(
        [
            "no credits/all paid",
            "all paid",
            "existing paid",
            "delayed previously",
            "critical/other existing credit",
        ],
        size=n,
        p=[0.05, 0.05, 0.53, 0.09, 0.28],
    )
    latent = (
        np.where(checking == "<0", 1.1, 0.0)
        + np.where(checking == "no checking", -0.8, 0.0)
        + (duration - 18) / 18
        + (amount - 2500) / 5000
        + np.where(history == "delayed previously", 0.45, 0.0)
        + rng.normal(0, 0.55, size=n)
    )
    probability = 1 / (1 + np.exp(-(latent - 0.15)))
    default = rng.random(n) < probability
    if int(default.sum()) < 12 or int((~default).sum()) < 12:
        default[:12] = True
        default[-12:] = False

    return pd.DataFrame(
        {
            "checking_status": checking,
            "duration": duration.astype(float),
            "credit_history": history,
            "purpose": rng.choice(
                ["radio/tv", "new car", "furniture/equipment", "business", "education"],
                size=n,
            ),
            "credit_amount": amount.astype(float),
            "savings_status": rng.choice(
                ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
                size=n,
            ),
            "employment": rng.choice(
                ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
                size=n,
            ),
            "installment_commitment": rng.integers(1, 5, size=n).astype(float),
            "personal_status": rng.choice(
                ["female div/dep/mar", "male single", "male mar/wid", "male div/sep"],
                size=n,
            ),
            "other_parties": rng.choice(["none", "co applicant", "guarantor"], size=n),
            "residence_since": rng.integers(1, 5, size=n).astype(float),
            "property_magnitude": rng.choice(
                ["real estate", "life insurance", "car", "no known property"],
                size=n,
            ),
            "age": rng.integers(19, 75, size=n).astype(float),
            "other_payment_plans": rng.choice(["bank", "stores", "none"], size=n),
            "housing": rng.choice(["rent", "own", "for free"], size=n),
            "existing_credits": rng.integers(1, 5, size=n).astype(float),
            "job": rng.choice(
                [
                    "unemp/unskilled non res",
                    "unskilled resident",
                    "skilled",
                    "high qualif/self emp/mgmt",
                ],
                size=n,
            ),
            "num_dependents": rng.integers(1, 3, size=n).astype(float),
            "own_telephone": rng.choice(["none", "yes"], size=n),
            "foreign_worker": rng.choice(["yes", "no"], size=n, p=[0.96, 0.04]),
            "class": np.where(default, "bad", "good"),
        }
    )
