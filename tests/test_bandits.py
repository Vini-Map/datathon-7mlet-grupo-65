"""Stage 3 tests for bandit policies and the simulator.

Built on a small toy environment (no raw CSV needed). They assert the contracts
the service and evaluation rely on: eligibility is always respected, cold-start
explores, results are reproducible, and the contextual policy beats `random`.
"""

from __future__ import annotations

import numpy as np
import pytest

from aep.bandits.base import Context
from aep.bandits.context_free import (
    UCB1,
    GreedyBaseline,
    NilosUCB,
    RandomPolicy,
    ThompsonSampling,
)
from aep.bandits.environment import build_environment
from aep.bandits.linucb import LinUCB
from aep.bandits.simulator import run_policy
from tests.test_synthetic import _toy_base

CONTEXT_FREE = [RandomPolicy, GreedyBaseline, ThompsonSampling, UCB1, NilosUCB]


def _ctx(eligible: list[int], n_arms: int = 8, dim: int = 11) -> Context:
    mask = np.zeros(n_arms, dtype=bool)
    mask[eligible] = True
    return Context(x=np.zeros(dim), eligible=mask)


@pytest.fixture(scope="module")
def env():
    return build_environment(n_steps=1500, base=_toy_base(n=400))


@pytest.mark.parametrize("cls", CONTEXT_FREE)
def test_context_free_respects_eligibility(cls) -> None:
    rng = np.random.default_rng(0)
    policy = cls(8, rng)
    eligible = [2, 5, 7]
    for _ in range(50):
        arm = policy.select(_ctx(eligible))
        assert arm in eligible
        policy.update(arm, 1.0, _ctx(eligible))


def test_linucb_respects_eligibility() -> None:
    policy = LinUCB(8, dim=11, rng=np.random.default_rng(0), alpha=0.5)
    eligible = [0, 3]
    for _ in range(30):
        arm = policy.select(_ctx(eligible))
        assert arm in eligible
        policy.update(arm, 1.0, _ctx(eligible))


def test_ucb_cold_start_tries_every_eligible_arm() -> None:
    policy = UCB1(8, np.random.default_rng(0))
    eligible = [1, 2, 4]
    picked = []
    for _ in range(len(eligible)):
        arm = policy.select(_ctx(eligible))
        policy.update(arm, 0.0, _ctx(eligible))
        picked.append(arm)
    # Unplayed arms have an infinite index, so the first picks cover every arm.
    assert set(picked) == set(eligible)


def test_run_policy_metrics_are_valid(env) -> None:
    res = run_policy(ThompsonSampling(env.n_arms, np.random.default_rng(1)), env)
    assert res.n_steps == env.n_steps
    assert 0.0 <= res.conversion_rate <= 1.0
    assert res.cum_regret >= 0.0
    assert 0.0 <= res.pct_optimal <= 1.0
    assert res.reward_curve.shape == (env.n_steps,)


def test_reproducible_same_seed(env) -> None:
    a = run_policy(ThompsonSampling(env.n_arms, np.random.default_rng(5)), env)
    b = run_policy(ThompsonSampling(env.n_arms, np.random.default_rng(5)), env)
    assert a.cum_reward == b.cum_reward
    assert a.cum_regret == b.cum_regret


def test_contextual_beats_random(env) -> None:
    rnd = run_policy(RandomPolicy(env.n_arms, np.random.default_rng(2)), env)
    lin = run_policy(LinUCB(env.n_arms, env.dim, np.random.default_rng(2), alpha=0.5), env)
    # LinUCB should achieve lower cumulative regret than uniform random.
    assert lin.cum_regret < rnd.cum_regret
