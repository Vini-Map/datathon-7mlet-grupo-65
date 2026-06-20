"""Promotion criteria — the automated gate before human approval (Stage 7).

A candidate must clear objective checks (no guardrail regressions, golden-set
quality, no material conversion drop, no fairness regression) before a human is
asked to approve it. Hard checks (adversarial guardrails) can never be waived.
"""

from __future__ import annotations

from dataclasses import dataclass

# Absolute floors and allowed regressions vs the current production policy.
MIN_GOLDEN_PASS_RATE = 0.85
MIN_ADVERSARIAL_PASS = 1.0  # hard: all suitability guardrails must pass
MAX_CONVERSION_DROP = 0.005  # candidate may not lose more than 0.5 pp vs prod
MAX_FAIRNESS_REGRESSION = 0.05  # disparity ratio may not drop more than this
MIN_FAIRNESS_FLOOR = 0.20


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    hard: bool = False


@dataclass
class PromotionDecision:
    passed: bool
    checks: list[Check]

    @property
    def hard_failures(self) -> list[Check]:
        return [c for c in self.checks if c.hard and not c.passed]


def evaluate_promotion(candidate: dict, production: dict | None) -> PromotionDecision:
    """Return a pass/fail decision with per-check detail.

    ``production`` is ``None`` for the very first policy (bootstrapping), in which
    case only absolute floors are checked.
    """
    checks: list[Check] = []

    adv = candidate.get("adversarial_pass", 0.0)
    checks.append(
        Check(
            "adversarial_guardrails",
            adv >= MIN_ADVERSARIAL_PASS,
            f"{adv:.0%} of adversarial cases pass (require {MIN_ADVERSARIAL_PASS:.0%})",
            hard=True,
        )
    )

    gpr = candidate.get("golden_pass_rate", 0.0)
    checks.append(
        Check(
            "golden_pass_rate",
            gpr >= MIN_GOLDEN_PASS_RATE,
            f"{gpr:.1%} (require >= {MIN_GOLDEN_PASS_RATE:.0%})",
        )
    )

    fair = candidate.get("fairness_disparity", 0.0)
    checks.append(
        Check(
            "fairness_floor",
            fair >= MIN_FAIRNESS_FLOOR,
            f"disparity {fair:.2f} (require >= {MIN_FAIRNESS_FLOOR:.2f})",
        )
    )

    if production is not None:
        conv_c, conv_p = candidate.get("conversion", 0.0), production.get("conversion", 0.0)
        checks.append(
            Check(
                "no_conversion_regression",
                conv_c >= conv_p - MAX_CONVERSION_DROP,
                f"candidate {conv_c:.2%} vs production {conv_p:.2%} "
                f"(allowed drop {MAX_CONVERSION_DROP:.2%})",
            )
        )
        fair_p = production.get("fairness_disparity", 0.0)
        checks.append(
            Check(
                "no_fairness_regression",
                fair >= fair_p - MAX_FAIRNESS_REGRESSION,
                f"candidate {fair:.2f} vs production {fair_p:.2f} "
                f"(allowed drop {MAX_FAIRNESS_REGRESSION:.2f})",
            )
        )

    return PromotionDecision(passed=all(c.passed for c in checks), checks=checks)
