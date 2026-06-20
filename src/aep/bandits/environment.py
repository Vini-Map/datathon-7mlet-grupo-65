"""Simulation environment for online bandit evaluation (Stage 3).

Wraps the Stage-2 oracle reward model and the factual base into a reproducible
stream of decision steps. For each step it precomputes the context vector, the
eligibility mask and the **true** reward probabilities of every arm (used to
measure expected regret). Reward realizations use one common-random-number
uniform per step (variance reduction across policies).

Delay is expressed in **decision steps** here (one impression per step), sampled
per chosen arm from a product-specific Poisson — this is what makes feedback
arrive late in the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aep.bandits.base import Context
from aep.data.loader import load_processed
from aep.synthetic.catalog import OFFER_IDS, catalog_frame, eligibility_mask
from aep.synthetic.config import SEED_EVENTS
from aep.synthetic.features import Standardizer, build_context_matrix, context_dim
from aep.synthetic.generate import _DELAY_LAMBDA
from aep.synthetic.reward_model import RewardModel


@dataclass
class Environment:
    X: np.ndarray  # (n_steps, dim)
    elig: np.ndarray  # (n_steps, n_arms) bool
    p: np.ndarray  # (n_steps, n_arms) true reward prob
    best_p: np.ndarray  # (n_steps,) best eligible reward prob
    best_arm: np.ndarray  # (n_steps,) argmax eligible arm
    u: np.ndarray  # (n_steps,) reward uniforms (CRN)
    arm_lambda: np.ndarray  # (n_arms,) mean delay per arm
    n_arms: int

    @property
    def n_steps(self) -> int:
        return self.X.shape[0]

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    def context(self, t: int) -> Context:
        return Context(x=self.X[t], eligible=self.elig[t])

    def reward(self, t: int, arm: int) -> float:
        """Realized Bernoulli reward for ``arm`` at step ``t`` (CRN draw)."""
        return float(self.u[t] < self.p[t, arm])

    def regret(self, t: int, arm: int) -> float:
        return float(self.best_p[t] - self.p[t, arm])


def build_environment(
    n_steps: int = 20_000,
    seed: int = SEED_EVENTS,
    base=None,
) -> Environment:
    base = load_processed() if base is None else base
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, len(base), size=n_steps)
    clients = base.iloc[idx].reset_index(drop=True)
    X, _ = build_context_matrix(clients, Standardizer.fit(base))
    subscribed = clients["subscribed"].to_numpy().astype(float)

    rm = RewardModel.build()
    p = rm.reward_prob(X, subscribed)  # (n_steps, n_arms)
    elig = eligibility_mask(clients)
    p_masked = np.where(elig, p, -1.0)
    best_arm = p_masked.argmax(axis=1)
    best_p = p_masked[np.arange(n_steps), best_arm]
    u = rng.random(n_steps)

    products = catalog_frame()["product"].to_numpy()
    arm_lambda = np.array([_DELAY_LAMBDA[prod] for prod in products])

    return Environment(
        X=X,
        elig=elig,
        p=p,
        best_p=best_p,
        best_arm=best_arm,
        u=u,
        arm_lambda=arm_lambda,
        n_arms=len(OFFER_IDS),
    )


def env_context_dim() -> int:
    return context_dim()
