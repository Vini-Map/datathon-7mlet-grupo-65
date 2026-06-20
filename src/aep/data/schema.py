"""Data dictionary as code.

Each :class:`ColumnSpec` documents one raw column: its kind, meaning, allowed
values and — crucially — whether it is **post-contact leakage** that must be
dropped before any modelling. Keeping this in code lets tests assert that the
processed base never carries a leakage column, and lets us export a Markdown
data dictionary that always matches the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ColumnKind(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TARGET = "target"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: ColumnKind
    description: str
    is_leakage: bool = False
    leakage_reason: str = ""
    categories: tuple[str, ...] = field(default_factory=tuple)


# Order matches the raw CSV header of bank-additional-full.csv.
COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("age", ColumnKind.NUMERIC, "Client age in years."),
    ColumnSpec(
        "job",
        ColumnKind.CATEGORICAL,
        "Type of job.",
        categories=(
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown",
        ),
    ),
    ColumnSpec(
        "marital",
        ColumnKind.CATEGORICAL,
        "Marital status.",
        categories=("divorced", "married", "single", "unknown"),
    ),
    ColumnSpec(
        "education",
        ColumnKind.CATEGORICAL,
        "Education level.",
        categories=(
            "basic.4y",
            "basic.6y",
            "basic.9y",
            "high.school",
            "illiterate",
            "professional.course",
            "university.degree",
            "unknown",
        ),
    ),
    ColumnSpec(
        "default",
        ColumnKind.CATEGORICAL,
        "Has credit in default?",
        categories=("no", "yes", "unknown"),
    ),
    ColumnSpec(
        "housing",
        ColumnKind.CATEGORICAL,
        "Has a housing loan?",
        categories=("no", "yes", "unknown"),
    ),
    ColumnSpec(
        "loan",
        ColumnKind.CATEGORICAL,
        "Has a personal loan?",
        categories=("no", "yes", "unknown"),
    ),
    ColumnSpec(
        "contact",
        ColumnKind.CATEGORICAL,
        "Contact communication type.",
        categories=("cellular", "telephone"),
    ),
    ColumnSpec(
        "month",
        ColumnKind.CATEGORICAL,
        "Last contact month of year.",
        categories=(
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ),
    ),
    ColumnSpec(
        "day_of_week",
        ColumnKind.CATEGORICAL,
        "Last contact day of the week.",
        categories=("mon", "tue", "wed", "thu", "fri"),
    ),
    ColumnSpec(
        "duration",
        ColumnKind.NUMERIC,
        "Last contact duration, in seconds.",
        is_leakage=True,
        leakage_reason=(
            "Known only AFTER the call ends, so it is unavailable at decision "
            "time. It almost perfectly encodes the outcome (duration=0 => y=no). "
            "UCI explicitly advises discarding it for realistic models."
        ),
    ),
    ColumnSpec(
        "campaign",
        ColumnKind.NUMERIC,
        "Number of contacts performed during this campaign for this client "
        "(includes the last contact).",
    ),
    ColumnSpec(
        "pdays",
        ColumnKind.NUMERIC,
        "Days since the client was last contacted in a previous campaign "
        "(999 = never previously contacted).",
    ),
    ColumnSpec(
        "previous",
        ColumnKind.NUMERIC,
        "Number of contacts performed before this campaign for this client.",
    ),
    ColumnSpec(
        "poutcome",
        ColumnKind.CATEGORICAL,
        "Outcome of the previous campaign.",
        categories=("failure", "nonexistent", "success"),
    ),
    ColumnSpec("emp.var.rate", ColumnKind.NUMERIC, "Employment variation rate (quarterly)."),
    ColumnSpec("cons.price.idx", ColumnKind.NUMERIC, "Consumer price index (monthly)."),
    ColumnSpec("cons.conf.idx", ColumnKind.NUMERIC, "Consumer confidence index (monthly)."),
    ColumnSpec("euribor3m", ColumnKind.NUMERIC, "Euribor 3-month rate (daily)."),
    ColumnSpec("nr.employed", ColumnKind.NUMERIC, "Number of employees (quarterly)."),
    ColumnSpec(
        "y",
        ColumnKind.TARGET,
        "Did the client subscribe a term deposit?",
        categories=("no", "yes"),
    ),
)

TARGET_COLUMN = "y"
TARGET_POSITIVE = "yes"

LEAKAGE_COLUMNS: tuple[str, ...] = tuple(c.name for c in COLUMNS if c.is_leakage)


def feature_columns(include_leakage: bool = False) -> list[str]:
    """Return feature column names (everything except the target)."""
    return [
        c.name
        for c in COLUMNS
        if c.kind is not ColumnKind.TARGET and (include_leakage or not c.is_leakage)
    ]


def column(name: str) -> ColumnSpec:
    for spec in COLUMNS:
        if spec.name == name:
            return spec
    raise KeyError(name)
