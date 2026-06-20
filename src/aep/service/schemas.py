"""Pydantic request/response contracts for the decision service (Stage 5).

`ClientContext` mirrors the processed-base schema with sensible defaults, so a
caller can POST a minimal JSON body and override only the fields that matter.
Only non-sensitive features are accepted — no identifiers, income or wealth.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClientContext(BaseModel):
    """Non-sensitive decision context (processed Bank Marketing schema)."""

    age: int = Field(40, ge=17, le=120)
    job: str = "admin."
    marital: str = "married"
    education: str = "university.degree"
    default: str = "no"
    housing: str = "yes"
    loan: str = "no"
    contact: str = "cellular"
    month: str = "may"
    day_of_week: str = "mon"
    campaign: int = Field(1, ge=1)
    pdays: int = 999
    previous: int = Field(0, ge=0)
    poutcome: str = "nonexistent"
    emp_var_rate: float = Field(1.1, alias="emp.var.rate")
    cons_price_idx: float = Field(93.994, alias="cons.price.idx")
    cons_conf_idx: float = Field(-36.4, alias="cons.conf.idx")
    euribor3m: float = 4.857
    nr_employed: float = Field(5191.0, alias="nr.employed")
    subscribed: int = Field(0, ge=0, le=1)

    model_config = {"populate_by_name": True}

    def to_features(self) -> dict:
        """Return a dict keyed by the raw processed-schema column names."""
        return {
            "age": self.age,
            "job": self.job,
            "marital": self.marital,
            "education": self.education,
            "default": self.default,
            "housing": self.housing,
            "loan": self.loan,
            "contact": self.contact,
            "month": self.month,
            "day_of_week": self.day_of_week,
            "campaign": self.campaign,
            "pdays": self.pdays,
            "previous": self.previous,
            "poutcome": self.poutcome,
            "emp.var.rate": self.emp_var_rate,
            "cons.price.idx": self.cons_price_idx,
            "cons.conf.idx": self.cons_conf_idx,
            "euribor3m": self.euribor3m,
            "nr.employed": self.nr_employed,
            "subscribed": self.subscribed,
        }


class DecisionResponse(BaseModel):
    audit_id: str
    timestamp: str
    offer_id: str
    arm_index: int
    score: float
    eligible_offers: list[str]
    scores: dict[str, float]
    reason_codes: list[str]
    policy_version: str


class ExplainResponse(BaseModel):
    decision: DecisionResponse
    explanation: str
    provider: str
    citations: list[str]


class SummaryResponse(BaseModel):
    summary: str
    provider: str


class PolicyQuery(BaseModel):
    query: str
    k: int = Field(3, ge=1, le=6)


class PolicyHit(BaseModel):
    doc_id: str
    title: str
    score: float
    text: str


class PolicyResponse(BaseModel):
    hits: list[PolicyHit]
