"""Build, simulate and compare bandit policies, logging to MLflow (Stage 3).

`run_comparison` runs every policy over the same environment stream (fair,
common-random-number comparison), logs each as an MLflow run with its
hyper-parameters and metrics, and returns the results. `sweep_nilos` traces the
confidence x exploration x conversion trade-off for Nilos-UCB.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aep.bandits.base import Policy
from aep.bandits.context_free import (
    UCB1,
    GreedyBaseline,
    NilosUCB,
    RandomPolicy,
    ThompsonSampling,
)
from aep.bandits.environment import Environment, build_environment
from aep.bandits.linucb import LinUCB
from aep.bandits.neural import NeuralBandit
from aep.bandits.simulator import SimResult, run_policy
from aep.mlops.tracking import start_run

PolicyBuilder = Callable[[np.random.Generator, int, int], Policy]

# Hyper-parameters are documented here and logged to MLflow per run.
POLICY_BUILDERS: dict[str, tuple[PolicyBuilder, dict]] = {
    "random": (lambda rng, k, d: RandomPolicy(k, rng), {}),
    "greedy_baseline": (lambda rng, k, d: GreedyBaseline(k, rng), {}),
    "thompson": (
        lambda rng, k, d: ThompsonSampling(k, rng, alpha0=1.0, beta0=1.0),
        {"alpha0": 1.0, "beta0": 1.0},
    ),
    "ucb1": (lambda rng, k, d: UCB1(k, rng), {}),
    "nilos_ucb": (lambda rng, k, d: NilosUCB(k, rng, c=1.0), {"c": 1.0}),
    "linucb": (lambda rng, k, d: LinUCB(k, d, rng, alpha=0.5), {"alpha": 0.5}),
    "neural": (
        lambda rng, k, d: NeuralBandit(k, d, rng, seed=0),
        {"hidden": 32, "lr": 0.01, "eps_decay": 0.9995},
    ),
}

CONTEXTUAL = {"linucb", "neural"}


def run_comparison(
    n_steps: int = 20_000,
    seed: int = 123,
    delayed: bool = True,
    log_mlflow: bool = True,
    env: Environment | None = None,
) -> dict[str, SimResult]:
    """Simulate all policies on the same environment; log to MLflow."""
    env = env or build_environment(n_steps=n_steps)
    results: dict[str, SimResult] = {}

    for name, (builder, params) in POLICY_BUILDERS.items():
        policy = builder(np.random.default_rng(seed), env.n_arms, env.dim)
        res = run_policy(policy, env, delayed=delayed)
        results[name] = res

        if log_mlflow:
            with start_run(run_name=name):
                import mlflow

                mlflow.log_params(
                    {
                        "policy": name,
                        "contextual": name in CONTEXTUAL,
                        "n_steps": env.n_steps,
                        "delayed": delayed,
                        "seed": seed,
                        **params,
                    }
                )
                mlflow.log_metrics(res.summary())
    return results


def sweep_nilos(
    cs: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 4.0),
    n_steps: int = 20_000,
    seed: int = 123,
    delayed: bool = True,
    env: Environment | None = None,
) -> dict[float, SimResult]:
    """Sweep the Nilos-UCB confidence coefficient ``c``."""
    env = env or build_environment(n_steps=n_steps)
    out: dict[float, SimResult] = {}
    for c in cs:
        policy = NilosUCB(env.n_arms, np.random.default_rng(seed), c=c)
        out[c] = run_policy(policy, env, delayed=delayed)
    return out
