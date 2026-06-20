"""LinUCB — disjoint linear contextual bandit (Stage 3).

Each arm keeps a ridge-regression estimate of E[reward | context] and an
upper-confidence bonus from the estimate's covariance:

    theta_a = A_a^{-1} b_a
    score_a = theta_a . x  +  alpha * sqrt(x^T A_a^{-1} x)

This lets the decision depend on **context**, so the policy can route each
segment to the arm that actually converts for it (the heterogeneity the EDA and
Stage-2 oracle show). ``alpha`` controls exploration. A constant 1 is appended
to the context for an intercept; cold-start is handled by the ridge prior
(A = I), which yields a large initial bonus.
"""

from __future__ import annotations

import numpy as np

from aep.bandits.base import Context, Policy


class LinUCB(Policy):
    name = "linucb"

    def __init__(
        self,
        n_arms: int,
        dim: int,
        rng: np.random.Generator | None = None,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(n_arms, rng)
        self.dim = dim + 1  # +1 for intercept
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self.A = np.stack([np.eye(self.dim) for _ in range(self.n_arms)])
        self.A_inv = np.stack([np.eye(self.dim) for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, self.dim))

    @staticmethod
    def _augment(x: np.ndarray) -> np.ndarray:
        return np.append(x, 1.0)

    def _scores(self, x: np.ndarray) -> np.ndarray:
        theta = np.einsum("aij,aj->ai", self.A_inv, self.b)
        mean = theta @ x
        bonus = self.alpha * np.sqrt(np.einsum("i,aij,j->a", x, self.A_inv, x))
        return mean + bonus

    def select(self, ctx: Context) -> int:
        x = self._augment(ctx.x)
        return self._argmax_eligible(self._scores(x), ctx.eligible)

    def update(self, arm: int, reward: float, ctx: Context) -> None:
        x = self._augment(ctx.x)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        # Sherman-Morrison rank-1 update of the inverse (cheaper than re-inverting).
        Ainv = self.A_inv[arm]
        Ax = Ainv @ x
        self.A_inv[arm] = Ainv - np.outer(Ax, Ax) / (1.0 + x @ Ax)
