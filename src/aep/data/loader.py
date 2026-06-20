"""Load and process the Bank Marketing base into a leakage-free table.

``build_processed`` is the single entry point used by the CLI and downstream
stages. It (1) reads the raw semicolon-separated CSV, (2) drops post-contact
leakage columns documented in :mod:`aep.data.schema`, (3) encodes the binary
target, and (4) writes a Parquet artifact under ``data/processed/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aep.config import REPO_ROOT
from aep.data.schema import (
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    TARGET_POSITIVE,
    feature_columns,
)
from aep.data.source import BANK_MARKETING, DatasetSource, verify_checksum

PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "bank_marketing.parquet"
TARGET_ENCODED = "subscribed"


class RawDataNotFoundError(FileNotFoundError):
    """Raised when the raw Kaggle CSV is missing, with download guidance."""


def _download_hint(source: DatasetSource) -> str:
    return (
        f"Raw file not found: {source.raw_path}\n"
        f"Download the Bank Marketing base first. Options:\n"
        f"  - Kaggle CLI:  kaggle datasets download -d {source.kaggle_ref} -p data/kaggle --unzip\n"
        f"  - UCI mirror:  {source.upstream_url}\n"
        f"    (unzip the nested archives and copy `bank-additional-full.csv` into data/kaggle/)"
    )


def load_raw(source: DatasetSource = BANK_MARKETING, verify: bool = True) -> pd.DataFrame:
    """Read the raw CSV exactly as distributed (no transformations)."""
    if not source.raw_path.exists():
        raise RawDataNotFoundError(_download_hint(source))
    if verify and not verify_checksum(source):
        raise ValueError(
            f"Checksum mismatch for {source.raw_path}. Expected {source.sha256}. "
            "The raw file differs from the pinned version."
        )
    return pd.read_csv(source.raw_path, sep=source.separator)


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage columns and encode the target. Returns a new frame."""
    out = df.copy()

    present_leakage = [c for c in LEAKAGE_COLUMNS if c in out.columns]
    out = out.drop(columns=present_leakage)

    # Encode binary target as 0/1 and drop the original string column.
    out[TARGET_ENCODED] = (out[TARGET_COLUMN] == TARGET_POSITIVE).astype(int)
    out = out.drop(columns=[TARGET_COLUMN])

    # Keep a stable, documented column order: features then encoded target.
    ordered = feature_columns(include_leakage=False) + [TARGET_ENCODED]
    return out[ordered]


def build_processed(
    source: DatasetSource = BANK_MARKETING,
    out_path: Path = PROCESSED_PATH,
    verify: bool = True,
) -> pd.DataFrame:
    """Full Stage-1 pipeline: load raw -> process -> persist Parquet."""
    processed = process(load_raw(source, verify=verify))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(out_path, index=False)
    return processed


def load_processed(out_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the persisted processed base, building it on demand if absent."""
    if not out_path.exists():
        return build_processed(out_path=out_path)
    return pd.read_parquet(out_path)
