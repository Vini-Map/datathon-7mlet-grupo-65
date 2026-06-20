"""Provenance of the factual reference dataset (Kaggle / UCI Bank Marketing).

Tracking source, version, license and checksum in code (not just prose) means
the pipeline can *verify* it is operating on the expected base, which is part of
the reproducibility and auditability requirements.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from aep.config import REPO_ROOT


@dataclass(frozen=True)
class DatasetSource:
    """Immutable description of an external dataset and how to obtain it."""

    name: str
    kaggle_ref: str
    upstream_url: str
    license_name: str
    license_url: str
    citation: str
    raw_filename: str
    sha256: str
    n_rows: int
    n_raw_columns: int
    separator: str = ";"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def raw_path(self) -> Path:
        return REPO_ROOT / "data" / "kaggle" / self.raw_filename


# The Kaggle dataset `henriqueyamahata/bank-marketing` redistributes the UCI
# "Bank Marketing" dataset (bank-additional-full.csv). We pin the UCI upstream
# because it is downloadable without credentials and is the canonical source.
BANK_MARKETING = DatasetSource(
    name="Bank Marketing (bank-additional-full)",
    kaggle_ref="henriqueyamahata/bank-marketing",
    upstream_url="https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    license_name="Creative Commons Attribution 4.0 International (CC BY 4.0)",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    citation=(
        "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C5K306"
    ),
    raw_filename="bank-additional-full.csv",
    sha256="74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8",
    n_rows=41188,
    n_raw_columns=21,
    notes=(
        "Telemarketing campaigns of a Portuguese bank (2008-2010).",
        "Target `y` = whether the client subscribed a term deposit.",
        "`duration` is post-contact leakage and is dropped (see schema).",
    ),
)


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(source: DatasetSource = BANK_MARKETING) -> bool:
    """Return True if the on-disk raw file matches the pinned checksum."""
    if not source.raw_path.exists():
        return False
    return sha256_of(source.raw_path) == source.sha256
