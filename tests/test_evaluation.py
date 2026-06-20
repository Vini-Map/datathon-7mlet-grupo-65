"""Stage 4 tests: golden-set logic and (when data is present) offline metrics.

The pass-criteria logic is unit-tested with a stub decision (no data needed).
Data-dependent integration tests are skipped without the raw Kaggle CSV.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aep.data.source import BANK_MARKETING
from aep.evaluation.golden import _check_case

RAW_PRESENT = BANK_MARKETING.raw_path.exists()


@dataclass
class _StubDecision:
    offer_id: str
    eligible_offers: list[str]
    arm_index: int = 0


def _case(criteria, expected_reward=0.1):
    return {"pass_criteria": criteria, "expected_reward": expected_reward}


def test_eligible_criterion() -> None:
    dec = _StubDecision("O3", ["O1", "O2"])  # O3 not eligible
    passed, fails = _check_case(_case([{"type": "eligible"}]), dec, 0.05)
    assert not passed and fails


def test_action_not_criterion() -> None:
    dec = _StubDecision("O8", ["O8", "O1"])
    passed, _ = _check_case(_case([{"type": "action_not", "actions": ["O8"]}]), dec, 0.05)
    assert not passed


def test_action_in_criterion() -> None:
    dec = _StubDecision("O1", ["O1", "O2"])
    passed, _ = _check_case(_case([{"type": "action_in", "actions": ["O1"]}]), dec, 0.05)
    assert passed


def test_reward_frac_criterion() -> None:
    dec = _StubDecision("O1", ["O1"])
    # chosen oracle reward 0.05 vs expected 0.10 -> frac 0.5 < 0.7 threshold.
    passed, _ = _check_case(
        _case([{"type": "reward_frac", "min_frac": 0.7}], expected_reward=0.10), dec, 0.05
    )
    assert not passed
    passed2, _ = _check_case(
        _case([{"type": "reward_frac", "min_frac": 0.7}], expected_reward=0.10), dec, 0.09
    )
    assert passed2


# --- integration (need the raw CSV to build the base/oracle) ----------------


@pytest.mark.skipif(not RAW_PRESENT, reason="raw Kaggle CSV not present")
def test_golden_set_has_at_least_20_cases() -> None:
    from aep.evaluation.golden import build_golden_set, load_golden

    build_golden_set()
    cases = load_golden()
    assert len(cases) >= 20
    required = {"case_id", "category", "context", "expected_action", "pass_criteria"}
    assert all(required <= set(c) for c in cases)


@pytest.mark.skipif(not RAW_PRESENT, reason="raw Kaggle CSV not present")
def test_adversarial_guardrails_all_pass() -> None:
    from aep.bandits.serving import ServingPolicy
    from aep.evaluation.golden import evaluate_golden

    res = evaluate_golden(ServingPolicy().fit())
    adv = res[res["category"] == "adversarial"]
    assert adv["passed"].all()


@pytest.mark.skipif(not RAW_PRESENT, reason="raw Kaggle CSV not present")
def test_offpolicy_estimates_track_oracle() -> None:
    from aep.evaluation.ope import evaluate_offpolicy

    ope = evaluate_offpolicy()
    # Target policy beats the uniform logging policy and approaches the oracle.
    assert ope.oracle_true_target > ope.logged_value
    assert ope.oracle_true_target <= ope.oracle_upper_bound + 1e-9
    # SNIPS lands within a reasonable band of the oracle-true value.
    assert abs(ope.snips - ope.oracle_true_target) < 0.03
