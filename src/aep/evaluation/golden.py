"""Golden set: versioned evaluation cases with explicit pass/fail (Stage 4).

Each case carries a full context, the oracle-optimal eligible action, the
expected reward and a machine-checkable pass criterion. Coverage spans typical
clients, every synthetic segment, edge cases (extreme/unknown values) and
adversarial scenarios (e.g. trying to elicit a suitability-restricted offer for
an ineligible client, or a reward-hacking context).

`build_golden_set` writes `data/golden_set/evaluation_cases.jsonl` (versioned);
`evaluate_golden` runs a policy against it and returns per-case pass/fail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aep.config import REPO_ROOT
from aep.data.loader import load_processed
from aep.synthetic.catalog import OFFER_IDS, eligibility_mask
from aep.synthetic.features import Standardizer, build_context_matrix
from aep.synthetic.reward_model import RewardModel

GOLDEN_PATH = REPO_ROOT / "data" / "golden_set" / "evaluation_cases.jsonl"
REWARD_FRAC_THRESHOLD = 0.70

# A realistic default context; cases override only the fields they care about.
_TEMPLATE: dict = {
    "age": 40,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
    "subscribed": 0,
}


def _row(**overrides) -> dict:
    return {**_TEMPLATE, **overrides}


def _oracle_probs(ctx: dict, std: Standardizer, rm: RewardModel) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame([ctx])
    x, _ = build_context_matrix(df, std)
    sub = df["subscribed"].to_numpy().astype(float)
    elig = eligibility_mask(df)[0]
    probs = rm.reward_prob(x, sub)[0]
    return probs, elig


@dataclass
class GoldenCase:
    case_id: str
    category: str
    context: dict
    expected_action: str
    expected_reward: float
    justification: str
    pass_criteria: list[dict]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def _make_case(
    case_id: str,
    category: str,
    ctx: dict,
    justification: str,
    std: Standardizer,
    rm: RewardModel,
    criteria: list[dict] | None = None,
) -> GoldenCase:
    probs, elig = _oracle_probs(ctx, std, rm)
    masked = np.where(elig, probs, -1.0)
    best = int(masked.argmax())
    expected = OFFER_IDS[best]
    if criteria is None:
        criteria = [{"type": "reward_frac", "min_frac": REWARD_FRAC_THRESHOLD}]
    return GoldenCase(
        case_id=case_id,
        category=category,
        context=ctx,
        expected_action=expected,
        expected_reward=round(float(masked[best]), 5),
        justification=justification,
        pass_criteria=criteria,
    )


def _cases(std: Standardizer, rm: RewardModel) -> list[GoldenCase]:
    cases: list[GoldenCase] = []

    # --- per-segment typical cases (covers all 6 synthetic segments) --------
    seg_specs = [
        ("young_wc", dict(age=28, job="management", education="university.degree")),
        ("young_nwc", dict(age=26, job="blue-collar", education="basic.9y")),
        ("mid_wc", dict(age=45, job="admin.", education="university.degree")),
        ("mid_nwc", dict(age=50, job="services", education="high.school")),
        ("senior_nwc", dict(age=67, job="retired", education="basic.4y")),
        ("senior_wc", dict(age=63, job="self-employed", education="professional.course")),
    ]
    for seg, ov in seg_specs:
        cases.append(
            _make_case(
                f"segment_{seg}",
                "segment",
                _row(**ov),
                f"Typical {seg} client; policy should route to a near-optimal eligible offer.",
                std,
                rm,
            )
        )

    # --- typical cases with previous-campaign / channel variation -----------
    cases.append(
        _make_case(
            "typical_prev_success",
            "typical",
            _row(age=42, poutcome="success", pdays=6, previous=2),
            "Prior successful contact strongly lifts term-deposit conversion.",
            std,
            rm,
        )
    )
    cases.append(
        _make_case(
            "typical_telephone_senior",
            "typical",
            _row(age=70, job="retired", contact="telephone"),
            "Senior on telephone channel; insurance/term-deposit tend to fit.",
            std,
            rm,
        )
    )
    cases.append(
        _make_case(
            "typical_app_young",
            "typical",
            _row(age=24, job="student", education="high.school", default="no"),
            "Young digital client; app-push offers engage best.",
            std,
            rm,
        )
    )

    cases.append(
        _make_case(
            "typical_high_euribor",
            "typical",
            _row(age=38, job="technician", euribor3m=5.0, **{"emp.var.rate": 1.4}),
            "High-rate macro regime; investment/term-deposit appeal shifts.",
            std,
            rm,
        )
    )
    cases.append(
        _make_case(
            "typical_with_loans",
            "typical",
            _row(age=48, housing="yes", loan="yes", job="blue-collar"),
            "Client already carrying loans; savings/term-deposit are reasonable.",
            std,
            rm,
        )
    )

    # --- edge cases ---------------------------------------------------------
    cases.append(
        _make_case(
            "edge_high_campaign",
            "edge",
            _row(campaign=56, previous=0),
            "Extreme contact fatigue (max campaign count).",
            std,
            rm,
        )
    )
    cases.append(
        _make_case(
            "edge_high_previous",
            "edge",
            _row(previous=7, pdays=3, poutcome="failure"),
            "Heavily re-contacted client with a prior failure.",
            std,
            rm,
        )
    )
    cases.append(
        _make_case(
            "edge_min_age",
            "edge",
            _row(age=18, job="student", education="basic.4y"),
            "Minimum plausible age; many age-gated offers ineligible.",
            std,
            rm,
        )
    )
    cases.append(
        _make_case(
            "edge_max_age",
            "edge",
            _row(age=95, job="retired"),
            "Very old client; loan offer ineligible (age cap).",
            std,
            rm,
        )
    )
    cases.append(
        _make_case(
            "edge_all_unknown",
            "edge",
            _row(
                job="unknown",
                education="unknown",
                marital="unknown",
                default="unknown",
                housing="unknown",
                loan="unknown",
            ),
            "Maximal missingness (all 'unknown'); policy must still decide safely.",
            std,
            rm,
        )
    )
    cases.append(
        _make_case(
            "edge_recession_macro",
            "edge",
            _row(
                **{
                    "emp.var.rate": -3.4,
                    "euribor3m": 0.634,
                    "cons.conf.idx": -50.8,
                    "nr.employed": 4963.6,
                }
            ),
            "Extreme recessionary macro context (min indicators).",
            std,
            rm,
        )
    )

    # --- adversarial scenarios ---------------------------------------------
    # Premium-bait: white-collar but default=yes -> O8 still requires age>=30 only,
    # but credit/loan/investment require no default. Policy must not pick a
    # no-default-only offer for a defaulting client.
    cases.append(
        _make_case(
            "adv_default_blocks_credit",
            "adversarial",
            _row(age=40, job="management", default="yes"),
            "Client in default: credit_card/personal_loan/investment are ineligible; "
            "the policy must never select them (suitability guardrail).",
            std,
            rm,
            criteria=[
                {"type": "eligible"},
                {"type": "action_not", "actions": ["O3", "O4", "O5"]},
            ],
        )
    )
    # Premium suitability: young non-white-collar must not get Premium Advisory.
    cases.append(
        _make_case(
            "adv_premium_for_ineligible",
            "adversarial",
            _row(age=22, job="blue-collar", education="basic.4y"),
            "Young blue-collar client is not suitable for Premium Advisory (O8); it "
            "must never be shown.",
            std,
            rm,
            criteria=[{"type": "eligible"}, {"type": "action_not", "actions": ["O8"]}],
        )
    )
    # Reward-hacking style: contradictory context that should not yield a high-risk
    # offer; require the chosen arm to be eligible and reasonably near-optimal.
    cases.append(
        _make_case(
            "adv_contradictory_context",
            "adversarial",
            _row(
                age=30, job="unknown", default="unknown", poutcome="success", pdays=999, previous=5
            ),
            "Contradictory signals (previous=5 but pdays=999 'never contacted'); the "
            "policy must stay within eligible arms and avoid pathological choices.",
            std,
            rm,
            criteria=[{"type": "eligible"}, {"type": "reward_frac", "min_frac": 0.50}],
        )
    )
    cases.append(
        _make_case(
            "adv_insurance_underage",
            "adversarial",
            _row(age=24, job="services"),
            "Insurance (O7) requires age >= 25; it must not be shown to a 24-year-old.",
            std,
            rm,
            criteria=[{"type": "eligible"}, {"type": "action_not", "actions": ["O7"]}],
        )
    )
    cases.append(
        _make_case(
            "adv_loan_overage",
            "adversarial",
            _row(age=70, job="retired"),
            "Personal Loan (O4) caps age at 65; it must not be shown to a 70-year-old.",
            std,
            rm,
            criteria=[{"type": "eligible"}, {"type": "action_not", "actions": ["O4"]}],
        )
    )

    return cases


def build_golden_set(path: Path = GOLDEN_PATH) -> Path:
    base = load_processed()
    std = Standardizer.fit(base)
    rm = RewardModel.build()
    cases = _cases(std, rm)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(c.to_json() + "\n")
    return path


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    if not path.exists():
        build_golden_set(path)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _check_case(case: dict, decision, oracle_chosen: float) -> tuple[bool, list[str]]:
    """Return (passed, failed_criteria_descriptions)."""
    failures: list[str] = []
    for crit in case["pass_criteria"]:
        ctype = crit["type"]
        if ctype == "eligible":
            if decision.offer_id not in decision.eligible_offers:
                failures.append("chose an ineligible offer")
        elif ctype == "action_in":
            if decision.offer_id not in crit["actions"]:
                failures.append(f"action {decision.offer_id} not in {crit['actions']}")
        elif ctype == "action_not":
            if decision.offer_id in crit["actions"]:
                failures.append(f"action {decision.offer_id} is forbidden")
        elif ctype == "reward_frac":
            if case["expected_reward"] <= 0:
                continue
            frac = oracle_chosen / case["expected_reward"]
            if frac < crit["min_frac"]:
                failures.append(f"reward frac {frac:.2f} < {crit['min_frac']}")
    return (len(failures) == 0, failures)


def evaluate_golden(policy, path: Path = GOLDEN_PATH) -> pd.DataFrame:
    """Run ``policy`` against the golden set; return a per-case result frame."""
    cases = load_golden(path)
    std = Standardizer.fit(load_processed())
    rm = RewardModel.build()

    rows = []
    for case in cases:
        df = pd.DataFrame([case["context"]])
        decision = policy.decide_row(df)
        probs, _ = _oracle_probs(case["context"], std, rm)
        oracle_chosen = float(probs[decision.arm_index])
        passed, failures = _check_case(case, decision, oracle_chosen)
        rows.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "expected_action": case["expected_action"],
                "chosen_action": decision.offer_id,
                "oracle_reward_chosen": round(oracle_chosen, 5),
                "expected_reward": case["expected_reward"],
                "passed": passed,
                "failures": "; ".join(failures),
            }
        )
    return pd.DataFrame(rows)
