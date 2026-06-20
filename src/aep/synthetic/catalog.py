"""Offer catalog — the bandit *arms* (Stage 2).

The catalog is **physically separate** from the Kaggle base: offers are an
invented commercial layer (product x channel x message tone) with explicit
**eligibility / suitability** rules. Eligibility is vectorized over the processed
base so an offer is only ever shown to clients it is allowed for — this both
models suitability and provides material for the exposure-fairness analysis
(Stage 4).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# An eligibility predicate maps the processed base to a boolean mask.
EligibilityFn = Callable[[pd.DataFrame], np.ndarray]


def _all_eligible(df: pd.DataFrame) -> np.ndarray:
    return np.ones(len(df), dtype=bool)


def _no_default(df: pd.DataFrame) -> np.ndarray:
    return df["default"].to_numpy() != "yes"


@dataclass(frozen=True)
class Offer:
    offer_id: str
    name: str
    product: str
    channel: str
    message_tone: str
    base_appeal: float  # latent baseline conversion logit (before context effects)
    eligibility: EligibilityFn
    eligibility_desc: str


# Hand-authored, interpretable catalog. base_appeal values are intentionally
# spread so no single arm dominates across all contexts; the context-dependent
# affinities live in the reward model.
OFFERS: tuple[Offer, ...] = (
    Offer(
        "O1",
        "Term Deposit — Standard",
        "term_deposit",
        "cellular",
        "informative",
        -2.1,
        _all_eligible,
        "All clients.",
    ),
    Offer(
        "O2",
        "Term Deposit — Social Proof",
        "term_deposit",
        "telephone",
        "social_proof",
        -2.3,
        _all_eligible,
        "All clients.",
    ),
    Offer(
        "O3",
        "Credit Card — Cashback",
        "credit_card",
        "app_push",
        "reward",
        -2.0,
        _no_default,
        "Clients without credit in default.",
    ),
    Offer(
        "O4",
        "Personal Loan — Limited Offer",
        "personal_loan",
        "cellular",
        "urgency",
        -2.4,
        lambda df: _no_default(df) & df["age"].between(21, 65).to_numpy(),
        "No default, age 21-65.",
    ),
    Offer(
        "O5",
        "Investment Fund",
        "investment_fund",
        "app_push",
        "informative",
        -2.6,
        lambda df: _no_default(df) & (df["age"].to_numpy() >= 30),
        "No default, age >= 30.",
    ),
    Offer(
        "O6",
        "Savings Account",
        "savings_account",
        "email",
        "informative",
        -1.9,
        _all_eligible,
        "All clients.",
    ),
    Offer(
        "O7",
        "Insurance — Protection",
        "insurance",
        "telephone",
        "reward",
        -2.5,
        lambda df: df["age"].to_numpy() >= 25,
        "Age >= 25.",
    ),
    Offer(
        "O8",
        "Premium Advisory",
        "premium_advisory",
        "app_push",
        "social_proof",
        -2.7,
        lambda df: (df["age"].to_numpy() >= 30)
        & np.isin(df["job"].to_numpy(), ["admin.", "management", "entrepreneur", "self-employed"]),
        "White-collar, age >= 30 (suitability-restricted).",
    ),
)

OFFER_IDS: tuple[str, ...] = tuple(o.offer_id for o in OFFERS)


def offer(offer_id: str) -> Offer:
    for o in OFFERS:
        if o.offer_id == offer_id:
            return o
    raise KeyError(offer_id)


def eligibility_mask(df: pd.DataFrame) -> np.ndarray:
    """Return a boolean matrix (n_rows x n_offers); True = offer eligible."""
    return np.column_stack([o.eligibility(df) for o in OFFERS])


def catalog_frame() -> pd.DataFrame:
    """Return the offer catalog as a DataFrame (the persisted artifact)."""
    return pd.DataFrame(
        {
            "offer_id": [o.offer_id for o in OFFERS],
            "name": [o.name for o in OFFERS],
            "product": [o.product for o in OFFERS],
            "channel": [o.channel for o in OFFERS],
            "message_tone": [o.message_tone for o in OFFERS],
            "base_appeal": [o.base_appeal for o in OFFERS],
            "eligibility": [o.eligibility_desc for o in OFFERS],
        }
    )
