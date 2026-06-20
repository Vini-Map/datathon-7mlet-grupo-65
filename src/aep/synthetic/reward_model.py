"""Ground-truth reward model for the synthetic experimentation layer.

This is the *oracle* the simulation samples from (Stage 3 bandits never see it).
It produces, per (client context, offer):

- ``p_click``  — immediate engagement (intermediate reward), driven by channel.
- ``p_conv``   — conversion given a click, driven by **context x offer affinity**
  plus the client's real receptiveness (the genuine Bank Marketing subscription
  signal), so the layer stays grounded in the factual base.

The structured affinities make the *optimal arm depend on context*, which is the
whole point of a contextual bandit. All randomness is seeded and the realized
parameter matrices are persisted to JSON for reproducibility/audit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aep.synthetic.catalog import OFFERS
from aep.synthetic.config import REWARD_PARAMS_PATH, SEED_REWARD_MODEL
from aep.synthetic.features import CONTEXT_FEATURES

_FIDX = {name: i for i, name in enumerate(CONTEXT_FEATURES)}

# Deterministic conversion-affinity boosts: offer_id -> {feature: weight}. These
# encode commercial intuition and guarantee segment heterogeneity.
_CONV_BOOSTS: dict[str, dict[str, float]] = {
    "O1": {"poutcome_success": 0.6, "contact_cellular": 0.5},
    "O2": {"poutcome_success": 0.9, "contact_cellular": -0.4, "age_senior": 0.4},
    "O3": {"contact_cellular": 0.6, "age": -0.5, "edu_university": 0.3},
    "O4": {"age": -0.8, "was_prev_contacted": 0.4, "white_collar": -0.4},
    "O5": {"age_senior": 0.8, "edu_university": 0.7, "white_collar": 0.6, "euribor3m": 0.4},
    "O6": {"age": 0.3, "poutcome_success": 0.3},
    "O7": {"age_senior": 0.6, "age": 0.4},
    "O8": {"white_collar": 1.0, "edu_university": 0.8, "age_senior": 0.5, "poutcome_success": 0.4},
}

# Immediate-engagement base logit per channel and a couple of context tilts.
_CHANNEL_CLICK_BASE: dict[str, float] = {
    "app_push": -0.4,
    "cellular": -0.7,
    "email": -1.2,
    "telephone": -1.0,
}
_CLICK_BOOSTS: dict[str, dict[str, float]] = {
    "app_push": {"age": -0.5, "edu_university": 0.3},
    "telephone": {"age": 0.4, "age_senior": 0.3},
    "cellular": {"age": -0.2},
    "email": {"edu_university": 0.3},
}

# Products aligned with the real subscription signal (deposit-like).
_RECEPTIVE_PRODUCTS = {"term_deposit", "savings_account", "investment_fund"}
_RECEPTIVENESS_GAMMA = 0.8

_P_CLICK_CLIP = (0.03, 0.75)
_P_CONV_CLIP = (0.01, 0.60)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class RewardModel:
    """Oracle reward model. Use :meth:`build` to construct (seeded)."""

    W: np.ndarray  # (n_offers, dim) conversion weights
    U: np.ndarray  # (n_offers, dim) click weights
    b_conv: np.ndarray  # (n_offers,) conversion bias (base_appeal + noise)
    b_click: np.ndarray  # (n_offers,) click bias (channel base)
    receptive: np.ndarray  # (n_offers,) gamma if product is deposit-like else 0
    feature_order: tuple[str, ...]

    @classmethod
    def build(cls, seed: int = SEED_REWARD_MODEL) -> RewardModel:
        rng = np.random.default_rng(seed)
        n, dim = len(OFFERS), len(CONTEXT_FEATURES)

        W = rng.normal(0.0, 0.25, size=(n, dim))
        U = rng.normal(0.0, 0.20, size=(n, dim))
        b_conv = np.empty(n)
        b_click = np.empty(n)
        receptive = np.zeros(n)

        for i, off in enumerate(OFFERS):
            for feat, val in _CONV_BOOSTS.get(off.offer_id, {}).items():
                W[i, _FIDX[feat]] += val
            for feat, val in _CLICK_BOOSTS.get(off.channel, {}).items():
                U[i, _FIDX[feat]] += val
            b_conv[i] = off.base_appeal + rng.normal(0.0, 0.10)
            b_click[i] = _CHANNEL_CLICK_BASE[off.channel]
            if off.product in _RECEPTIVE_PRODUCTS:
                receptive[i] = _RECEPTIVENESS_GAMMA

        return cls(W, U, b_conv, b_click, receptive, tuple(CONTEXT_FEATURES))

    # --- probabilities ------------------------------------------------------

    def click_prob(self, X: np.ndarray) -> np.ndarray:
        """(n_samples, n_offers) immediate click probability."""
        logits = self.b_click[None, :] + X @ self.U.T
        return np.clip(_sigmoid(logits), *_P_CLICK_CLIP)

    def conv_prob(self, X: np.ndarray, subscribed: np.ndarray) -> np.ndarray:
        """(n_samples, n_offers) conversion-given-click probability."""
        logits = self.b_conv[None, :] + X @ self.W.T
        logits = logits + subscribed[:, None] * self.receptive[None, :]
        return np.clip(_sigmoid(logits), *_P_CONV_CLIP)

    def reward_prob(self, X: np.ndarray, subscribed: np.ndarray) -> np.ndarray:
        """(n_samples, n_offers) P(conversion) = P(click) * P(conv|click)."""
        return self.click_prob(X) * self.conv_prob(X, subscribed)

    # --- persistence --------------------------------------------------------

    def to_json(self, path: Path = REWARD_PARAMS_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_order": list(self.feature_order),
            "offer_ids": [o.offer_id for o in OFFERS],
            "W": self.W.tolist(),
            "U": self.U.tolist(),
            "b_conv": self.b_conv.tolist(),
            "b_click": self.b_click.tolist(),
            "receptive": self.receptive.tolist(),
            "gamma": _RECEPTIVENESS_GAMMA,
            "p_click_clip": _P_CLICK_CLIP,
            "p_conv_clip": _P_CONV_CLIP,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
