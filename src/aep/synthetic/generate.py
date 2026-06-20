"""Generate the synthetic logged-bandit dataset (Stage 2).

Produces three artifacts, all seeded:

- ``offer_catalog``   — the arms (from :mod:`aep.synthetic.catalog`).
- ``offer_events``    — one logged impression per row: client context reference,
  synthetic segment, the **logged action** (offer shown), its **propensity**
  under the logging policy, and the immediate click reward.
- ``delayed_rewards`` — the conversion outcome per event with a sampled delay and
  the day it becomes observable (a non-converter is censored at the horizon).

The logging policy is **uniform over eligible offers**, so propensities are exact
(`1 / n_eligible`). That makes the dataset suitable for unbiased off-policy
evaluation (IPS) in Stage 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aep.data.loader import load_processed
from aep.synthetic.catalog import OFFER_IDS, catalog_frame, eligibility_mask
from aep.synthetic.config import (
    CENSOR_WINDOW_DAYS,
    DELAYED_REWARDS_PATH,
    HORIZON_DAYS,
    N_EVENTS,
    OFFER_CATALOG_PATH,
    OFFER_EVENTS_PATH,
    SEED_DELAY,
    SEED_EVENTS,
    STANDARDIZER_PATH,
)
from aep.synthetic.features import Standardizer, build_context_matrix
from aep.synthetic.reward_model import RewardModel

# Mean conversion delay (days) by product — deposit-like products settle slower.
_DELAY_LAMBDA: dict[str, float] = {
    "term_deposit": 6.0,
    "savings_account": 3.0,
    "investment_fund": 7.0,
    "credit_card": 2.0,
    "personal_loan": 4.0,
    "insurance": 5.0,
    "premium_advisory": 8.0,
}


def _segment(df: pd.DataFrame) -> np.ndarray:
    """Synthetic fairness segment: age band x white-collar status."""
    age = df["age"].to_numpy()
    band = np.where(age < 35, "young", np.where(age < 60, "mid", "senior"))
    white = np.isin(df["job"].to_numpy(), ["admin.", "management", "entrepreneur", "self-employed"])
    collar = np.where(white, "wc", "nwc")
    return np.char.add(np.char.add(band, "_"), collar)


def generate(
    n_events: int = N_EVENTS,
    horizon: int = HORIZON_DAYS,
    seed_events: int = SEED_EVENTS,
    seed_delay: int = SEED_DELAY,
    base: pd.DataFrame | None = None,
    persist: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate the synthetic dataset and return the frames.

    ``base`` defaults to the processed Bank Marketing table; pass a frame to run
    without the raw CSV (e.g. tests). ``persist=False`` skips writing artifacts.
    """
    base = load_processed() if base is None else base
    rng = np.random.default_rng(seed_events)

    # Sample client impressions (with replacement) from the factual base.
    idx = rng.integers(0, len(base), size=n_events)
    clients = base.iloc[idx].reset_index(drop=True)

    standardizer = Standardizer.fit(base)
    X, _ = build_context_matrix(clients, standardizer)
    subscribed = clients["subscribed"].to_numpy().astype(float)

    elig = eligibility_mask(clients)  # (n_events, n_offers)
    n_eligible = elig.sum(axis=1)

    # Logging policy: uniform over eligible offers (exact propensities).
    probs = elig / n_eligible[:, None]
    chosen = np.array([rng.choice(len(OFFER_IDS), p=probs[i]) for i in range(n_events)])
    propensity = 1.0 / n_eligible

    reward_model = RewardModel.build()
    p_click_all = reward_model.click_prob(X)
    p_conv_all = reward_model.conv_prob(X, subscribed)
    rows = np.arange(n_events)
    p_click = p_click_all[rows, chosen]
    p_conv = p_conv_all[rows, chosen]

    click = (rng.random(n_events) < p_click).astype(int)
    conv_given_click = (rng.random(n_events) < p_conv).astype(int)
    conversion = click * conv_given_click

    day = rng.integers(0, horizon, size=n_events)

    # Delayed-reward realization (separate seeded stream).
    rng_d = np.random.default_rng(seed_delay)
    product_by_idx = catalog_frame()["product"].to_numpy()  # aligned with OFFER_IDS
    lambdas = np.array([_DELAY_LAMBDA[product_by_idx[c]] for c in chosen])
    delay = 1 + rng_d.poisson(lambdas)
    # Converters realize after the delay; non-converters are censored at horizon.
    reward_available_day = np.where(conversion == 1, day + delay, day + CENSOR_WINDOW_DAYS)
    censored = conversion == 0

    events = pd.DataFrame(
        {
            "event_id": np.arange(n_events),
            "day": day,
            "client_idx": idx,
            "segment": _segment(clients),
            "subscribed": clients["subscribed"].to_numpy(),
            "offer_id": [OFFER_IDS[c] for c in chosen],
            "n_eligible": n_eligible,
            "propensity": propensity,
            "click": click,
        }
    )
    delayed = pd.DataFrame(
        {
            "event_id": np.arange(n_events),
            "conversion": conversion,
            "reward_delay_days": np.where(conversion == 1, delay, CENSOR_WINDOW_DAYS),
            "reward_available_day": reward_available_day,
            "censored": censored,
        }
    )
    catalog = catalog_frame()

    if persist:
        _persist(catalog, events, delayed, standardizer)
    return {"offer_catalog": catalog, "offer_events": events, "delayed_rewards": delayed}


def _persist(
    catalog: pd.DataFrame,
    events: pd.DataFrame,
    delayed: pd.DataFrame,
    standardizer: Standardizer,
) -> None:
    OFFER_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(OFFER_CATALOG_PATH, index=False)
    events.to_parquet(OFFER_EVENTS_PATH, index=False)
    delayed.to_parquet(DELAYED_REWARDS_PATH, index=False)
    RewardModel.build().to_json()
    Path(STANDARDIZER_PATH).write_text(
        json.dumps({"mean": standardizer.mean, "std": standardizer.std}, indent=2),
        encoding="utf-8",
    )


def load_synthetic() -> dict[str, pd.DataFrame]:
    """Load the persisted synthetic frames, generating them on demand."""
    paths = {
        "offer_catalog": OFFER_CATALOG_PATH,
        "offer_events": OFFER_EVENTS_PATH,
        "delayed_rewards": DELAYED_REWARDS_PATH,
    }
    if not all(p.exists() for p in paths.values()):
        return generate()
    return {k: pd.read_parquet(p) for k, p in paths.items()}
