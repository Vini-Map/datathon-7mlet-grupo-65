"""Context-free bandit policies (Stage 3).

Implemented from scratch:

- :class:`RandomPolicy` — uniform over eligible arms (exploration floor).
- :class:`GreedyBaseline` — **deterministic baseline**: one warm-up pull per arm
  then pure exploitation of the best historical arm.
- :class:`ThompsonSampling` — Beta-Bernoulli posterior sampling (documented priors).
- :class:`UCB1` — the classic upper-confidence-bound index.
- :class:`NilosUCB` — a confidence-parameterized UCB variant that exposes the
  confidence x exploration x conversion trade-off through a single coefficient.

All keep per-arm counts/successes and respect eligibility.
"""

from __future__ import annotations

import numpy as np

from aep.bandits.base import Context, Policy


class _ArmStats(Policy):
    """Shared per-arm success/count bookkeeping for context-free policies."""

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None) -> None:
        super().__init__(n_arms, rng)
        self.reset()

    def reset(self) -> None:
        self.counts = np.zeros(self.n_arms)
        self.successes = np.zeros(self.n_arms)

    @property
    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, self.successes / self.counts, 0.0)

    def update(self, arm: int, reward: float, ctx: Context) -> None:
        self.counts[arm] += 1
        self.successes[arm] += reward


class RandomPolicy(_ArmStats):
    name = "random"

    def select(self, ctx: Context) -> int:
        return int(self.rng.choice(ctx.eligible_idx))


class GreedyBaseline(_ArmStats):
    """Deterministic baseline: warm-up one pull per arm, then exploit."""

    name = "greedy_baseline"

    def select(self, ctx: Context) -> int:
        # Warm-up: play any eligible arm not yet tried (lowest index = deterministic).
        unplayed = ctx.eligible_idx[self.counts[ctx.eligible_idx] == 0]
        if unplayed.size > 0:
            return int(unplayed[0])
        # Exploit the best empirical mean (deterministic tie-break: lowest index).
        masked = np.where(ctx.eligible, self.means, -np.inf)
        return int(np.argmax(masked))


class ThompsonSampling(_ArmStats):
    """Beta-Bernoulli Thompson Sampling.

    Prior Beta(alpha0, beta0); default Beta(1, 1) = uniform (uninformative), so
    cold-start arms are sampled optimistically-but-fairly and exploration is
    driven by posterior uncertainty.
    """

    name = "thompson"

    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator | None = None,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        self.alpha0 = alpha0
        self.beta0 = beta0
        super().__init__(n_arms, rng)

    def select(self, ctx: Context) -> int:
        alpha = self.alpha0 + self.successes
        beta = self.beta0 + (self.counts - self.successes)
        theta = self.rng.beta(alpha, beta)
        return self._argmax_eligible(theta, ctx.eligible)


class UCB1(_ArmStats):
    """Classic UCB1: index = mean + sqrt(2 ln t / n). Unplayed arms get +inf."""

    name = "ucb1"

    def select(self, ctx: Context) -> int:
        t = self.counts.sum() + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = np.sqrt(2.0 * np.log(t) / self.counts)
        index = np.where(self.counts > 0, self.means + bonus, np.inf)
        return self._argmax_eligible(index, ctx.eligible)


class NilosUCB(_ArmStats):
    """Confidence-parameterized UCB.

    index = mean + c * sqrt(ln t / n)

    The coefficient ``c`` tunes the confidence level: small ``c`` favors
    exploitation (higher short-term conversion, risk of locking onto a
    sub-optimal arm), large ``c`` widens confidence bounds (more exploration,
    lower regret asymptotically). Sweeping ``c`` traces the
    confidence x exploration x conversion trade-off (see the Stage 3 report).
    UCB1 is the special case c = sqrt(2).
    """

    name = "nilos_ucb"

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None, c: float = 1.0) -> None:
        self.c = c
        super().__init__(n_arms, rng)

    def select(self, ctx: Context) -> int:
        t = self.counts.sum() + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = self.c * np.sqrt(np.log(t) / self.counts)
        index = np.where(self.counts > 0, self.means + bonus, np.inf)
        return self._argmax_eligible(index, ctx.eligible)
