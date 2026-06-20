"""Stage 2 tests for the synthetic enrichment layer.

These run without the raw Kaggle CSV by passing a small toy base to the
generator, and assert the contracts the downstream bandits/evaluation rely on:
seeded reproducibility, valid probabilities, exact propensities, and the
funnel constraint (conversion implies a click).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aep.synthetic.catalog import OFFER_IDS, OFFERS, catalog_frame, eligibility_mask, offer
from aep.synthetic.features import CONTEXT_FEATURES, build_context_matrix, context_dim
from aep.synthetic.generate import generate
from aep.synthetic.reward_model import RewardModel


def _toy_base(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "job": rng.choice(["admin.", "blue-collar", "technician", "management"], n),
            "education": rng.choice(["university.degree", "basic.9y", "high.school"], n),
            "default": rng.choice(["no", "unknown", "yes"], n, p=[0.8, 0.18, 0.02]),
            "contact": rng.choice(["cellular", "telephone"], n),
            "poutcome": rng.choice(["nonexistent", "failure", "success"], n),
            "previous": rng.integers(0, 4, n),
            "pdays": rng.choice([999, 3, 6], n, p=[0.9, 0.05, 0.05]),
            "emp.var.rate": rng.normal(0, 1, n),
            "euribor3m": rng.uniform(0.6, 5.0, n),
            "cons.conf.idx": rng.normal(-40, 4, n),
            "subscribed": rng.integers(0, 2, n),
        }
    )


# --- catalog ----------------------------------------------------------------


def test_catalog_has_eight_unique_arms() -> None:
    assert len(OFFERS) == 8
    assert len(set(OFFER_IDS)) == 8
    assert list(catalog_frame()["offer_id"]) == list(OFFER_IDS)


def test_premium_offer_is_suitability_restricted() -> None:
    base = _toy_base()
    mask = eligibility_mask(base)
    premium_col = OFFER_IDS.index("O8")
    # O8 requires white-collar & age>=30, so it cannot be eligible for everyone.
    assert mask[:, premium_col].sum() < len(base)
    # O1 is open to all.
    assert mask[:, OFFER_IDS.index("O1")].all()


def test_offer_lookup() -> None:
    assert offer("O5").product == "investment_fund"


# --- features ---------------------------------------------------------------


def test_context_matrix_shape() -> None:
    base = _toy_base()
    X, std = build_context_matrix(base)
    assert X.shape == (len(base), context_dim())
    assert X.shape[1] == len(CONTEXT_FEATURES)
    assert set(std.mean) and set(std.std)


# --- reward model -----------------------------------------------------------


def test_reward_model_is_deterministic() -> None:
    a, b = RewardModel.build(seed=123), RewardModel.build(seed=123)
    assert np.allclose(a.W, b.W) and np.allclose(a.U, b.U)


def test_reward_probabilities_are_valid() -> None:
    base = _toy_base()
    X, _ = build_context_matrix(base)
    sub = base["subscribed"].to_numpy().astype(float)
    rm = RewardModel.build()
    pc, pv = rm.click_prob(X), rm.conv_prob(X, sub)
    assert pc.shape == (len(base), len(OFFERS))
    assert ((pc >= 0) & (pc <= 1)).all()
    assert ((pv >= 0) & (pv <= 1)).all()
    # Reward prob is the product of the two stages.
    assert np.allclose(rm.reward_prob(X, sub), pc * pv)


# --- generation -------------------------------------------------------------


def test_generate_contracts() -> None:
    out = generate(n_events=500, base=_toy_base(), persist=False)
    events, delayed = out["offer_events"], out["delayed_rewards"]
    assert len(events) == len(delayed) == 500

    # Exact propensity under uniform-over-eligible logging.
    assert np.allclose(events["propensity"], 1.0 / events["n_eligible"])

    # Funnel: every conversion must have a click.
    joined = events.merge(delayed, on="event_id")
    assert (joined.loc[joined["conversion"] == 1, "click"] == 1).all()

    # Non-converters are censored; converters are not.
    assert (delayed.loc[delayed["conversion"] == 0, "censored"]).all()
    assert not (delayed.loc[delayed["conversion"] == 1, "censored"]).any()


def test_generate_is_reproducible() -> None:
    base = _toy_base()
    a = generate(n_events=300, base=base, persist=False)["offer_events"]
    b = generate(n_events=300, base=base, persist=False)["offer_events"]
    pd.testing.assert_frame_equal(a, b)
