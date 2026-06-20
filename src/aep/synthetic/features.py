"""Context featurization shared by the reward model (Stage 2) and the
contextual bandits (Stage 3).

A client row from the processed Bank Marketing base is mapped to a small, fixed,
interpretable context vector. Numeric features are standardized using stats
computed from the base (returned so they can be persisted/reused), and a handful
of binary segment indicators capture the heterogeneity the EDA surfaced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Numeric features kept as context (all known at decision time, no leakage).
NUMERIC_FEATURES: tuple[str, ...] = (
    "age",
    "previous",
    "emp.var.rate",
    "euribor3m",
    "cons.conf.idx",
)

# White-collar job grouping used for one segment indicator.
_WHITE_COLLAR = {"admin.", "management", "entrepreneur", "self-employed"}

# Final context feature order (numeric standardized first, then binary segments).
CONTEXT_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + (
    "was_prev_contacted",
    "poutcome_success",
    "contact_cellular",
    "edu_university",
    "white_collar",
    "age_senior",
)


@dataclass(frozen=True)
class Standardizer:
    """Mean/std for the numeric context features (persisted for reproducibility)."""

    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> Standardizer:
        mean = {c: float(df[c].mean()) for c in NUMERIC_FEATURES}
        std = {c: float(df[c].std(ddof=0)) or 1.0 for c in NUMERIC_FEATURES}
        return cls(mean=mean, std=std)


def build_context_matrix(
    df: pd.DataFrame, standardizer: Standardizer | None = None
) -> tuple[np.ndarray, Standardizer]:
    """Return (X, standardizer) where X has columns :data:`CONTEXT_FEATURES`."""
    std = standardizer or Standardizer.fit(df)

    cols: list[np.ndarray] = []
    for c in NUMERIC_FEATURES:
        cols.append((df[c].to_numpy(dtype=float) - std.mean[c]) / std.std[c])

    cols.append((df["pdays"].to_numpy() != 999).astype(float))
    cols.append((df["poutcome"].to_numpy() == "success").astype(float))
    cols.append((df["contact"].to_numpy() == "cellular").astype(float))
    cols.append((df["education"].to_numpy() == "university.degree").astype(float))
    cols.append(np.isin(df["job"].to_numpy(), list(_WHITE_COLLAR)).astype(float))
    cols.append((df["age"].to_numpy() >= 60).astype(float))

    X = np.column_stack(cols)
    return X, std


def context_dim() -> int:
    return len(CONTEXT_FEATURES)
