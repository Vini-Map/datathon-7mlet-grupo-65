"""Stage 7 tests: registry stages + approval gate, promotion criteria, PSI.

These run without the raw CSV (registry/promotion/PSI are pure logic). The
data-dependent drift monitor and lifecycle demo are exercised in the report.
"""

from __future__ import annotations

import numpy as np
import pytest

from aep.mlops.drift import psi
from aep.mlops.promotion import evaluate_promotion
from aep.mlops.registry import ApprovalError, PolicyRegistry

_GOOD = {
    "conversion": 0.14,
    "golden_pass_rate": 0.91,
    "adversarial_pass": 1.0,
    "fairness_disparity": 0.30,
}


@pytest.fixture
def registry(tmp_path):
    return PolicyRegistry(path=tmp_path / "reg.json")


# --- registry / approval gate ----------------------------------------------


def test_register_starts_in_staging(registry) -> None:
    pv = registry.register("v1", _GOOD, {"alpha": 0.5})
    assert pv.stage == "staging"
    assert registry.production() is None


def test_promote_requires_approval(registry) -> None:
    registry.register("v1", _GOOD, {"alpha": 0.5})
    with pytest.raises(ApprovalError):
        registry.promote("v1")


def test_approve_then_promote(registry) -> None:
    registry.register("v1", _GOOD, {"alpha": 0.5})
    registry.approve("v1", approver="alice", notes="ok")
    registry.promote("v1")
    assert registry.production().version == "v1"
    assert registry.production().approver == "alice"


def test_promotion_archives_incumbent_and_rolls_back(registry) -> None:
    for v in ("v1", "v2"):
        registry.register(v, _GOOD, {"alpha": 0.5})
        registry.approve(v, approver="alice")
    registry.promote("v1")
    registry.promote("v2")
    assert registry.production().version == "v2"
    rolled = registry.rollback()
    assert rolled.version == "v1"
    assert registry.production().version == "v1"


def test_registry_persists_across_instances(tmp_path) -> None:
    path = tmp_path / "reg.json"
    reg = PolicyRegistry(path=path)
    reg.register("v1", _GOOD, {})
    reg.approve("v1", approver="a")
    reg.promote("v1")
    assert PolicyRegistry(path=path).production().version == "v1"


# --- promotion gate ---------------------------------------------------------


def test_promotion_passes_for_good_candidate() -> None:
    assert evaluate_promotion(_GOOD, production=None).passed


def test_promotion_hard_fails_on_adversarial_breach() -> None:
    bad = {**_GOOD, "adversarial_pass": 0.9}
    decision = evaluate_promotion(bad, production=None)
    assert not decision.passed
    assert decision.hard_failures


def test_promotion_blocks_conversion_regression() -> None:
    prod = {**_GOOD, "conversion": 0.14}
    candidate = {**_GOOD, "conversion": 0.10}  # big drop
    assert not evaluate_promotion(candidate, production=prod).passed


# --- PSI --------------------------------------------------------------------


def test_psi_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x.copy()) < 1e-6


def test_psi_large_for_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(3, 1, 5000)
    assert psi(ref, cur) > 0.25
