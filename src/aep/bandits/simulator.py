"""Online bandit simulator with delayed feedback (Stage 3).

Runs a policy over the environment's decision stream, optionally delivering each
reward several steps after the decision (delayed feedback). Reports the metrics
the datathon asks for: realized **reward**, **regret**, **exploration** and
simulated **conversion**.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from aep.bandits.base import Policy
from aep.bandits.environment import Environment


@dataclass
class SimResult:
    policy: str
    n_steps: int
    cum_reward: float
    conversion_rate: float
    cum_regret: float
    avg_regret: float
    pct_optimal: float
    exploration_entropy: float
    arm_counts: dict[int, int]
    reward_curve: np.ndarray = field(repr=False)
    regret_curve: np.ndarray = field(repr=False)

    def summary(self) -> dict[str, float]:
        return {
            "cum_reward": self.cum_reward,
            "conversion_rate": self.conversion_rate,
            "cum_regret": self.cum_regret,
            "avg_regret": self.avg_regret,
            "pct_optimal": self.pct_optimal,
            "exploration_entropy": self.exploration_entropy,
        }


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    h = -(p * np.log(p)).sum()
    return float(h / np.log(len(counts))) if len(counts) > 1 else 0.0


def run_policy(
    policy: Policy,
    env: Environment,
    delayed: bool = True,
    delay_seed: int = 7,
) -> SimResult:
    """Simulate ``policy`` over ``env``; return metrics and learning curves."""
    rng_delay = np.random.default_rng(delay_seed)
    pending: dict[int, list[tuple[int, float, object]]] = defaultdict(list)

    chosen = np.empty(env.n_steps, dtype=int)
    rewards = np.empty(env.n_steps)
    regrets = np.empty(env.n_steps)

    for t in range(env.n_steps):
        # Deliver any rewards that have matured by now.
        if delayed and t in pending:
            for arm, rew, ctx in pending.pop(t):
                policy.update(arm, rew, ctx)

        ctx = env.context(t)
        arm = policy.select(ctx)
        rew = env.reward(t, arm)

        chosen[t] = arm
        rewards[t] = rew
        regrets[t] = env.regret(t, arm)

        if delayed:
            delay = 1 + int(rng_delay.poisson(env.arm_lambda[arm]))
            pending[t + delay].append((arm, rew, ctx))
        else:
            policy.update(arm, rew, ctx)

    # Flush any rewards still pending after the last step.
    for bucket in pending.values():
        for arm, rew, ctx in bucket:
            policy.update(arm, rew, ctx)

    counts = np.bincount(chosen, minlength=env.n_arms)
    pct_optimal = float((chosen == env.best_arm).mean())
    return SimResult(
        policy=policy.name,
        n_steps=env.n_steps,
        cum_reward=float(rewards.sum()),
        conversion_rate=float(rewards.mean()),
        cum_regret=float(regrets.sum()),
        avg_regret=float(regrets.mean()),
        pct_optimal=pct_optimal,
        exploration_entropy=_entropy(counts),
        arm_counts={i: int(c) for i, c in enumerate(counts)},
        reward_curve=np.cumsum(rewards),
        regret_curve=np.cumsum(regrets),
    )
