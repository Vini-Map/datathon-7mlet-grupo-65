"""Synthetic commercial-policy and suitability documents for RAG (Stage 5).

These are **invented** policy snippets — not real bank rules — that mirror the
eligibility/suitability logic encoded in the offer catalog. The RAG layer
retrieves them so the assistant can ground its explanations in "internal policy"
without exposing any real or sensitive content.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDoc:
    doc_id: str
    title: str
    text: str


POLICY_DOCS: tuple[PolicyDoc, ...] = (
    PolicyDoc(
        "POL-001",
        "Offer eligibility & suitability",
        "Clients with credit in default must not be offered credit cards, personal "
        "loans or investment funds. The Premium Advisory offer is restricted to "
        "white-collar clients (admin, management, entrepreneur, self-employed) aged "
        "30 or older. Personal Loan is limited to ages 21-65. Insurance requires age "
        "25 or older. Term deposit and savings have no eligibility restriction.",
    ),
    PolicyDoc(
        "POL-002",
        "Channel policy",
        "Channels are app push, cellular call, telephone call and email. Younger, "
        "digitally-active clients engage more on app push; older clients respond "
        "better on telephone. Email is low-intrusion but low-engagement. Channel "
        "choice is part of the offer definition and is logged with every decision.",
    ),
    PolicyDoc(
        "POL-003",
        "Exploration & experimentation policy",
        "Offer selection uses multi-armed bandits balancing exploration and "
        "exploitation rather than fixed rules or long A/B tests. New offers start "
        "cold and are explored under uncertainty. Exploration must respect "
        "eligibility at all times; an ineligible offer is never shown regardless of "
        "its estimated reward.",
    ),
    PolicyDoc(
        "POL-004",
        "Fairness & exposure policy",
        "Offer exposure is monitored across client segments. No single offer should "
        "dominate exposure without review, and large gaps in delivered value between "
        "segments must be investigated. Segments here are synthetic operational "
        "constructs (age band and occupation group), not protected attributes.",
    ),
    PolicyDoc(
        "POL-005",
        "Reward, conversion and delayed feedback",
        "The immediate reward is a click on the offer; the final reward is a "
        "conversion (product subscription), which can arrive days after the "
        "decision. Policies must tolerate delayed and censored feedback and avoid "
        "over-committing to an arm before its reward is observed.",
    ),
    PolicyDoc(
        "POL-006",
        "Data minimization & privacy (LGPD)",
        "Decisions use only non-sensitive context features. No client identifiers, "
        "income, wealth, gender or race are used. Decision logs retain reason codes, "
        "the selected offer and the policy version for auditability, under a defined "
        "retention period. Sensitive decisions keep a human in the loop.",
    ),
)


def doc_by_id(doc_id: str) -> PolicyDoc:
    for d in POLICY_DOCS:
        if d.doc_id == doc_id:
            return d
    raise KeyError(doc_id)
