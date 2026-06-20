"""Stage 5 tests: API contract, decision, audit logging and the assistant.

The assistant tests use the offline MockProvider (no API key). API tests use the
FastAPI TestClient. Data-dependent decision tests are skipped without the raw
Kaggle CSV; the assistant/RAG and schema tests always run.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aep.assistant.assistant import Assistant
from aep.assistant.provider import MockProvider, get_provider
from aep.assistant.rag import Retriever
from aep.data.source import BANK_MARKETING
from aep.service.app import app
from aep.service.schemas import ClientContext

RAW_PRESENT = BANK_MARKETING.raw_path.exists()
client = TestClient(app)


# --- schema / contract ------------------------------------------------------


def test_client_context_aliases_and_defaults() -> None:
    ctx = ClientContext(age=30, **{"emp.var.rate": -1.0})
    feats = ctx.to_features()
    assert feats["age"] == 30
    assert feats["emp.var.rate"] == -1.0
    assert feats["job"] == "admin."  # default


def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_decide_validation_rejects_bad_age() -> None:
    resp = client.post("/decide", json={"age": 5})
    assert resp.status_code == 422  # below ge=17


# --- RAG / assistant (offline mock) ----------------------------------------


def test_retriever_finds_relevant_policy() -> None:
    hits = Retriever().retrieve("client in default credit card eligibility", k=2)
    assert hits and hits[0].score > 0
    # POL-001 covers default/eligibility.
    assert any(h.doc.doc_id == "POL-001" for h in hits)


def test_mock_provider_is_default_and_offline() -> None:
    assert isinstance(get_provider("mock"), MockProvider)
    out = get_provider("mock").complete("sys", "hello world")
    assert "mock-llm" in out


def test_assistant_summarize_uses_mock() -> None:
    ans = Assistant(provider=MockProvider()).summarize_experiment("conversion +133% lift")
    assert ans.provider == "mock"
    assert "133%" in ans.text


# --- decision + audit (need data) ------------------------------------------


@pytest.mark.skipif(not RAW_PRESENT, reason="raw Kaggle CSV not present")
def test_decide_returns_eligible_audited_offer() -> None:
    resp = client.post("/decide", json={"age": 40, "job": "management", "default": "no"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["offer_id"] in body["eligible_offers"]
    assert body["policy_version"]
    assert body["audit_id"]
    assert body["reason_codes"]


@pytest.mark.skipif(not RAW_PRESENT, reason="raw Kaggle CSV not present")
def test_decide_respects_default_suitability_guardrail() -> None:
    # A defaulting client must never be offered credit/loan/investment (O3/O4/O5).
    resp = client.post("/decide", json={"age": 40, "job": "management", "default": "yes"})
    assert resp.status_code == 200
    assert resp.json()["offer_id"] not in {"O3", "O4", "O5"}


@pytest.mark.skipif(not RAW_PRESENT, reason="raw Kaggle CSV not present")
def test_explain_endpoint_grounds_in_policy() -> None:
    resp = client.post("/assistant/explain", json={"age": 40, "job": "management"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "mock"
    assert body["citations"]
    assert body["decision"]["offer_id"]
