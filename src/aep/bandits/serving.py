"""A frozen, deterministic decision policy for evaluation and serving.

The online LinUCB from Stage 3 is fit by **replaying the logged events** and then
frozen: at decision time it scores arms by the learned mean only (no exploration
bonus), so decisions are deterministic and reproducible — exactly what the golden
set (Stage 4) and the service (Stage 5) need.

It also exposes the per-arm scores and a small set of reason codes, which the
service turns into an auditable decision record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aep.bandits.linucb import LinUCB
from aep.data.loader import load_processed
from aep.synthetic.catalog import OFFER_IDS, eligibility_mask, offer
from aep.synthetic.features import Standardizer, build_context_matrix
from aep.synthetic.generate import load_synthetic

POLICY_VERSION = "linucb-v1"


@dataclass(frozen=True)
class Decision:
    offer_id: str
    arm_index: int
    score: float
    eligible_offers: list[str]
    scores: dict[str, float]
    reason_codes: list[str]
    policy_version: str = POLICY_VERSION


class ServingPolicy:
    """Frozen LinUCB decision function fit from the synthetic logged events."""

    def __init__(self, alpha: float = 0.5, version: str = POLICY_VERSION) -> None:
        self.alpha = alpha
        self.version = version
        self._policy: LinUCB | None = None
        self._std: Standardizer | None = None

    # --- fitting ------------------------------------------------------------

    def fit(self) -> ServingPolicy:
        base = load_processed()
        self._std = Standardizer.fit(base)
        events = load_synthetic()["offer_events"]
        delayed = load_synthetic()["delayed_rewards"]
        rewards = events.merge(delayed[["event_id", "conversion"]], on="event_id")

        clients = base.iloc[rewards["client_idx"].to_numpy()].reset_index(drop=True)
        X, _ = build_context_matrix(clients, self._std)
        elig = eligibility_mask(clients)
        arm_of = {oid: i for i, oid in enumerate(OFFER_IDS)}

        policy = LinUCB(len(OFFER_IDS), dim=X.shape[1], alpha=self.alpha)
        from aep.bandits.base import Context

        for i, row in enumerate(rewards.itertuples(index=False)):
            arm = arm_of[row.offer_id]
            ctx = Context(x=X[i], eligible=elig[i])
            policy.update(arm, float(row.conversion), ctx)
        self._policy = policy
        return self

    def _ensure_fit(self) -> None:
        if self._policy is None or self._std is None:
            self.fit()

    # --- decision -----------------------------------------------------------

    def _scores(self, x: np.ndarray) -> np.ndarray:
        """Exploitation scores: learned mean only (no exploration bonus)."""
        assert self._policy is not None
        xa = np.append(x, 1.0)
        theta = np.einsum("aij,aj->ai", self._policy.A_inv, self._policy.b)
        return theta @ xa

    def scores_matrix(self, X: np.ndarray) -> np.ndarray:
        """Exploitation scores for a (n, dim) context matrix -> (n, n_arms)."""
        self._ensure_fit()
        assert self._policy is not None
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])
        theta = np.einsum("aij,aj->ai", self._policy.A_inv, self._policy.b)
        return Xa @ theta.T

    def decide_indices(self, X: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        """Vectorized deterministic decisions; returns arm indices (n,)."""
        scores = self.scores_matrix(X)
        masked = np.where(eligible, scores, -np.inf)
        return masked.argmax(axis=1)

    def decide_row(self, row: pd.DataFrame) -> Decision:
        """Decide for a single-row DataFrame of raw (processed-schema) features."""
        self._ensure_fit()
        x, _ = build_context_matrix(row, self._std)
        eligible = eligibility_mask(row)[0]
        scores = self._scores(x[0])
        masked = np.where(eligible, scores, -np.inf)
        arm = int(np.argmax(masked))

        eligible_ids = [OFFER_IDS[i] for i in np.flatnonzero(eligible)]
        reason = [
            f"selected_max_expected_reward:{OFFER_IDS[arm]}",
            f"eligible_count:{len(eligible_ids)}",
            f"product:{offer(OFFER_IDS[arm]).product}",
            f"channel:{offer(OFFER_IDS[arm]).channel}",
        ]
        return Decision(
            offer_id=OFFER_IDS[arm],
            arm_index=arm,
            score=float(scores[arm]),
            eligible_offers=eligible_ids,
            scores={OFFER_IDS[i]: float(scores[i]) for i in range(len(OFFER_IDS))},
            reason_codes=reason,
            policy_version=self.version,
        )
