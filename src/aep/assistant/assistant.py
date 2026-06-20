"""The LLM + RAG assistant (Stage 5).

Three capabilities, all **grounded** in retrieved synthetic policy and in the
actual decision/experiment facts (never free-form about clients):

- :meth:`explain_decision` — why a given offer was chosen, citing policy.
- :meth:`retrieve_policy`   — surface relevant internal (synthetic) policy.
- :meth:`summarize_experiment` — plain-language summary of experiment metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aep.assistant.provider import LLMProvider, get_provider
from aep.assistant.rag import RetrievedDoc, Retriever
from aep.config import REPO_ROOT

SYSTEM_PROMPT = (
    "You are a compliance-aware assistant for a regulated offer-decisioning "
    "platform. Answer ONLY from the provided policy excerpts and decision facts. "
    "Be concise and factual. Never invent client data, income or identifiers. "
    "Always note the relevant suitability/eligibility guardrail."
)


@dataclass
class AssistantAnswer:
    text: str
    provider: str
    citations: list[str] = field(default_factory=list)


class Assistant:
    def __init__(
        self, provider: LLMProvider | None = None, retriever: Retriever | None = None
    ) -> None:
        self.provider = provider or get_provider()
        self.retriever = retriever or Retriever()

    # --- policy retrieval ---------------------------------------------------

    def retrieve_policy(self, query: str, k: int = 3) -> list[RetrievedDoc]:
        return self.retriever.retrieve(query, k=k)

    # --- decision explanation ----------------------------------------------

    def explain_decision(self, decision, context: dict) -> AssistantAnswer:
        query = (
            f"eligibility suitability {decision.offer_id} "
            f"{' '.join(decision.reason_codes)} channel product"
        )
        hits = self.retrieve_policy(query, k=3)
        policy_text = "\n".join(f"[{h.doc.doc_id}] {h.doc.title}: {h.doc.text}" for h in hits)
        ctx_brief = ", ".join(
            f"{k}={context[k]}"
            for k in ("age", "job", "default", "contact", "poutcome")
            if k in context
        )
        prompt = (
            "Explain, for an internal reviewer, why the following offer was chosen.\n\n"
            f"DECISION FACTS:\n"
            f"- selected_offer: {decision.offer_id}\n"
            f"- eligible_offers: {decision.eligible_offers}\n"
            f"- reason_codes: {decision.reason_codes}\n"
            f"- policy_version: {decision.policy_version}\n\n"
            f"CLIENT CONTEXT (non-sensitive): {ctx_brief}\n\n"
            f"RELEVANT INTERNAL POLICY:\n{policy_text}\n\n"
            "Write 3-5 sentences: why this offer, which eligibility/suitability "
            "guardrail applied, and one caveat."
        )
        text = self.provider.complete(SYSTEM_PROMPT, prompt)
        return AssistantAnswer(
            text=text, provider=self.provider.name, citations=[h.doc.doc_id for h in hits]
        )

    # --- experiment summary -------------------------------------------------

    def summarize_experiment(self, metrics_text: str | None = None) -> AssistantAnswer:
        snapshot = metrics_text or _experiment_snapshot()
        prompt = (
            "Summarize the following offline experiment results for a business "
            "stakeholder in 3-4 sentences. State the recommended policy, its lift "
            "over the baseline, and one risk/limitation.\n\n"
            f"EXPERIMENT METRICS:\n{snapshot}"
        )
        text = self.provider.complete(SYSTEM_PROMPT, prompt)
        return AssistantAnswer(text=text, provider=self.provider.name)


def _experiment_snapshot() -> str:
    """Pull headline sections from the generated reports, if present."""
    parts: list[str] = []
    for rel in ("reports/bandit-comparison.md", "reports/evaluation.md"):
        path = REPO_ROOT / rel
        if path.exists():
            head = path.read_text(encoding="utf-8").split("## 2.")[0]
            parts.append(head.strip())
    return "\n\n".join(parts) if parts else "No experiment reports found yet."
