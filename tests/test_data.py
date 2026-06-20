"""Stage 1 tests for the data layer.

The raw Kaggle CSV is git-ignored, so tests that need it are skipped when it is
absent (e.g. CI/fresh clone). The leakage-removal contract is tested against a
tiny in-memory frame so it always runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aep.data import schema
from aep.data.loader import TARGET_ENCODED, process
from aep.data.source import BANK_MARKETING, verify_checksum

RAW_PRESENT = BANK_MARKETING.raw_path.exists()


def _toy_raw() -> pd.DataFrame:
    """A 2-row frame with every raw column, matching the documented schema."""
    return pd.DataFrame(
        {
            "age": [40, 55],
            "job": ["admin.", "technician"],
            "marital": ["married", "single"],
            "education": ["university.degree", "basic.9y"],
            "default": ["no", "unknown"],
            "housing": ["yes", "no"],
            "loan": ["no", "no"],
            "contact": ["cellular", "telephone"],
            "month": ["may", "jun"],
            "day_of_week": ["mon", "tue"],
            "duration": [261, 0],  # leakage column
            "campaign": [1, 2],
            "pdays": [999, 3],
            "previous": [0, 1],
            "poutcome": ["nonexistent", "success"],
            "emp.var.rate": [1.1, -0.1],
            "cons.price.idx": [93.994, 93.2],
            "cons.conf.idx": [-36.4, -42.0],
            "euribor3m": [4.857, 1.313],
            "nr.employed": [5191.0, 5099.1],
            "y": ["no", "yes"],
        }
    )


# --- schema -----------------------------------------------------------------


def test_duration_is_the_only_leakage_column() -> None:
    assert schema.LEAKAGE_COLUMNS == ("duration",)


def test_feature_columns_exclude_leakage_and_target() -> None:
    feats = schema.feature_columns(include_leakage=False)
    assert "duration" not in feats
    assert schema.TARGET_COLUMN not in feats
    assert "age" in feats and "job" in feats
    assert len(feats) == 19  # 20 raw inputs minus duration


def test_every_column_has_a_description() -> None:
    assert all(c.description for c in schema.COLUMNS)


# --- processing contract ----------------------------------------------------


def test_process_drops_leakage_and_encodes_target() -> None:
    out = process(_toy_raw())
    assert "duration" not in out.columns
    assert schema.TARGET_COLUMN not in out.columns
    assert TARGET_ENCODED in out.columns
    assert out[TARGET_ENCODED].tolist() == [0, 1]


def test_process_is_leakage_free_and_ordered() -> None:
    out = process(_toy_raw())
    expected = schema.feature_columns(include_leakage=False) + [TARGET_ENCODED]
    assert list(out.columns) == expected


# --- real-data checks (skipped when the raw file is absent) ------------------


@pytest.mark.skipif(not RAW_PRESENT, reason="raw Kaggle CSV not present")
def test_raw_checksum_matches_pinned() -> None:
    assert verify_checksum(BANK_MARKETING)


@pytest.mark.skipif(not RAW_PRESENT, reason="raw Kaggle CSV not present")
def test_real_processed_shape() -> None:
    from aep.data.loader import build_processed

    df = build_processed()
    assert len(df) == BANK_MARKETING.n_rows
    assert "duration" not in df.columns
    assert df[TARGET_ENCODED].isin({0, 1}).all()
