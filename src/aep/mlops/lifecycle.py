"""Policy lifecycle orchestration (Stage 7).

Ties the pieces together: compute a candidate policy's metrics from the offline
evaluation, run the promotion gate, and drive the registry through the full
experiment -> staging -> approval -> production -> rollback story that the banca
asks us to demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass

from aep.bandits.serving import ServingPolicy
from aep.evaluation.fairness import evaluate_fairness
from aep.evaluation.golden import evaluate_golden
from aep.evaluation.ope import evaluate_offpolicy
from aep.mlops.promotion import evaluate_promotion
from aep.mlops.registry import PolicyRegistry


def build_candidate_metrics(version: str, alpha: float) -> dict:
    """Fit a serving policy and compute its offline metrics for the registry."""
    policy = ServingPolicy(alpha=alpha, version=version).fit()
    golden = evaluate_golden(policy)
    ope = evaluate_offpolicy(policy)
    fair = evaluate_fairness(policy)
    adv = golden[golden["category"] == "adversarial"]["passed"].mean()
    return {
        "conversion": round(ope.oracle_true_target, 5),
        "snips": round(ope.snips, 5),
        "golden_pass_rate": round(float(golden["passed"].mean()), 4),
        "adversarial_pass": round(float(adv), 4),
        "fairness_disparity": round(fair.value_disparity_ratio, 4),
    }


@dataclass
class LifecycleStep:
    action: str
    detail: str


def run_lifecycle_demo(registry: PolicyRegistry) -> list[LifecycleStep]:
    """Demonstrate a new hypothesis going from experiment to controlled production.

    v1 (alpha=0.5) is the incumbent; v2 (alpha=1.0, "more exploration") is the new
    hypothesis. It is registered, gated, approved and promoted, then rolled back.
    """
    log: list[LifecycleStep] = []

    # 1. Incumbent: register, bootstrap-approve and promote.
    m1 = build_candidate_metrics("linucb-v1", alpha=0.5)
    registry.register("linucb-v1", m1, {"alpha": 0.5})
    decision1 = evaluate_promotion(m1, production=None)
    registry.approve("linucb-v1", approver="bootstrap", notes="initial production policy")
    registry.promote("linucb-v1")
    log.append(
        LifecycleStep(
            "register+promote v1", f"bootstrap gate passed={decision1.passed}; v1 -> production"
        )
    )

    # 2. New hypothesis v2: register as staging candidate.
    m2 = build_candidate_metrics("linucb-v2", alpha=1.0)
    registry.register("linucb-v2", m2, {"alpha": 1.0})
    log.append(LifecycleStep("register v2 (staging)", f"candidate metrics: {m2}"))

    # 3. Automated promotion gate vs current production.
    prod = registry.production()
    decision2 = evaluate_promotion(m2, production=prod.metrics if prod else None)
    log.append(
        LifecycleStep(
            "promotion gate v2",
            f"passed={decision2.passed}; "
            + "; ".join(f"{c.name}={'OK' if c.passed else 'FAIL'}" for c in decision2.checks),
        )
    )

    # 4. Human approval gate + promotion (only if the gate passed).
    if decision2.passed:
        registry.approve(
            "linucb-v2", approver="ml-reviewer", notes="metrics within bounds; guardrails intact"
        )
        registry.promote("linucb-v2")
        log.append(LifecycleStep("approve+promote v2", "v2 -> production, v1 -> archived"))
    else:
        log.append(LifecycleStep("v2 blocked", "gate failed; v2 stays in staging"))

    # 5. Demonstrate rollback to the previous production version.
    if registry.production() and registry.production().version == "linucb-v2":
        rolled = registry.rollback()
        log.append(LifecycleStep("rollback", f"production reverted to {rolled.version}"))

    return log
