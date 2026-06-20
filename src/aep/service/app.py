"""FastAPI decision service + LLM/RAG assistant endpoints (Stage 5).

Endpoints:
- ``GET  /health``               liveness + policy version.
- ``POST /decide``               context -> offer decision (audited).
- ``POST /assistant/explain``    context -> decision + grounded explanation.
- ``POST /assistant/summarize``  plain-language experiment summary.
- ``POST /assistant/policy``     retrieve relevant synthetic policy (RAG).
- ``GET  /audit/recent``         recent auditable decision records.

The serving policy and assistant are built lazily and cached. A missing raw base
yields a 503 with guidance rather than a 500.
"""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException

from aep import __version__
from aep.assistant.assistant import Assistant
from aep.bandits.serving import ServingPolicy
from aep.data.loader import RawDataNotFoundError
from aep.service import audit
from aep.service.schemas import (
    ClientContext,
    DecisionResponse,
    ExplainResponse,
    PolicyHit,
    PolicyQuery,
    PolicyResponse,
    SummaryResponse,
)

app = FastAPI(
    title="AEP — Adaptive Experimentation Platform",
    version=__version__,
    description="Bandit-based offer decisioning with an LLM+RAG assistant (Datathon 7MLET, Grupo 65).",
)

_policy: ServingPolicy | None = None
_assistant: Assistant | None = None


def get_policy() -> ServingPolicy:
    global _policy
    if _policy is None:
        try:
            _policy = ServingPolicy().fit()
        except RawDataNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Serving policy unavailable: {exc}. Run `aep data build` and "
                "`aep synth generate` first.",
            ) from exc
    return _policy


def get_assistant() -> Assistant:
    global _assistant
    if _assistant is None:
        _assistant = Assistant()
    return _assistant


def _decide(ctx: ClientContext):
    features = ctx.to_features()
    decision = get_policy().decide_row(pd.DataFrame([features]))
    record = audit.record_decision(features, decision)
    return decision, record, features


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "policy_version": ServingPolicy().version}


@app.post("/decide", response_model=DecisionResponse)
def decide(ctx: ClientContext) -> DecisionResponse:
    decision, record, _ = _decide(ctx)
    return DecisionResponse(
        audit_id=record["audit_id"],
        timestamp=record["timestamp"],
        offer_id=decision.offer_id,
        arm_index=decision.arm_index,
        score=decision.score,
        eligible_offers=decision.eligible_offers,
        scores=decision.scores,
        reason_codes=decision.reason_codes,
        policy_version=decision.policy_version,
    )


@app.post("/assistant/explain", response_model=ExplainResponse)
def explain(ctx: ClientContext) -> ExplainResponse:
    decision, record, features = _decide(ctx)
    answer = get_assistant().explain_decision(decision, features)
    return ExplainResponse(
        decision=DecisionResponse(
            audit_id=record["audit_id"],
            timestamp=record["timestamp"],
            offer_id=decision.offer_id,
            arm_index=decision.arm_index,
            score=decision.score,
            eligible_offers=decision.eligible_offers,
            scores=decision.scores,
            reason_codes=decision.reason_codes,
            policy_version=decision.policy_version,
        ),
        explanation=answer.text,
        provider=answer.provider,
        citations=answer.citations,
    )


@app.post("/assistant/summarize", response_model=SummaryResponse)
def summarize() -> SummaryResponse:
    answer = get_assistant().summarize_experiment()
    return SummaryResponse(summary=answer.text, provider=answer.provider)


@app.post("/assistant/policy", response_model=PolicyResponse)
def policy(query: PolicyQuery) -> PolicyResponse:
    hits = get_assistant().retrieve_policy(query.query, k=query.k)
    return PolicyResponse(
        hits=[
            PolicyHit(doc_id=h.doc.doc_id, title=h.doc.title, score=h.score, text=h.doc.text)
            for h in hits
        ]
    )


@app.get("/audit/recent")
def audit_recent(n: int = 10) -> dict:
    return {"records": audit.read_recent(n)}
