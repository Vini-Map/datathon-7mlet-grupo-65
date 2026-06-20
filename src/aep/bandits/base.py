"""Bandit policy interface and the per-decision context (Stage 3).

Every policy implements the same small contract so the simulator, the service
and the evaluation can treat them interchangeably:

- :meth:`Policy.select` chooses an **eligible** arm given a :class:`Context`.
- :meth:`Policy.update` ingests a (possibly delayed) reward for a past decision.

Context-free policies use only per-arm reward statistics; contextual policies
use the context vector ``x``. All policies must respect the eligibility mask.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Context:
    """A single decision context."""

    x: np.ndarray  # context feature vector, shape (dim,)
    eligible: np.ndarray  # boolean mask over all arms, shape (n_arms,)

    @property
    def eligible_idx(self) -> np.ndarray:
        return np.flatnonzero(self.eligible)


class Policy(ABC):
    """Base class for all bandit policies."""

    name: str = "policy"

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None) -> None:
        self.n_arms = n_arms
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def select(self, ctx: Context) -> int:
        """Return the index of the chosen (eligible) arm."""

    @abstractmethod
    def update(self, arm: int, reward: float, ctx: Context) -> None:
        """Incorporate an observed reward for ``arm`` under ``ctx``."""

    def reset(self) -> None:  # noqa: B027  # optional no-op hook, overridden when stateful
        """Clear learned state (optional)."""

    # --- helpers -------------------------------------------------------------

    def _argmax_eligible(self, scores: np.ndarray, eligible: np.ndarray) -> int:
        """Argmax of ``scores`` restricted to eligible arms (random tie-break)."""
        masked = np.where(eligible, scores, -np.inf)
        best = np.flatnonzero(masked == masked.max())
        return int(best[0] if best.size == 1 else self.rng.choice(best))
