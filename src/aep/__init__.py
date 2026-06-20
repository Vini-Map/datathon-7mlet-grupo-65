"""Adaptive Experimentation Platform (AEP).

End-to-end multi-armed-bandit offer-decisioning platform with an LLM+RAG
assistant, built for the 7MLET / POSTECH Datathon (Grupo 65).

The package is organized by stage of the project plan (see ``CLAUDE.md``):

- :mod:`aep.data`        — Kaggle loaders, data dictionary, source/version/license tracking.
- :mod:`aep.synthetic`   — seeded generators for the synthetic experimentation layer.
- :mod:`aep.bandits`     — baseline, Thompson Sampling, Nilos-UCB, LinUCB / neural bandit.
- :mod:`aep.evaluation` — offline evaluation, metrics, fairness.
- :mod:`aep.service`     — FastAPI + CLI, reason codes, auditable decision log.
- :mod:`aep.assistant`   — pluggable LLMProvider + RAG.
- :mod:`aep.mlops`       — MLflow tracking, drift, promotion/rollback.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
