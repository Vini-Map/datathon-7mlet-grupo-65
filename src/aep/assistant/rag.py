"""Retrieval over the synthetic policy documents (Stage 5).

A dependency-light TF-IDF + cosine retriever over :data:`POLICY_DOCS`. It is
deterministic and needs no external service, which keeps the demo reproducible.
In the Azure target architecture this maps to **Azure AI Search** (or a FAISS/
Chroma vector store) behind the same :class:`Retriever` interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from aep.assistant.documents import POLICY_DOCS, PolicyDoc


@dataclass(frozen=True)
class RetrievedDoc:
    doc: PolicyDoc
    score: float


class Retriever:
    """TF-IDF cosine retriever over the synthetic policy corpus."""

    def __init__(self, docs: tuple[PolicyDoc, ...] = POLICY_DOCS) -> None:
        self.docs = docs
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [f"{d.title}. {d.text}" for d in docs]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedDoc]:
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [RetrievedDoc(doc=self.docs[i], score=float(sims[i])) for i in order]
