"""Generate Markdown artifacts from the data layer.

Both the data dictionary and the data-quality report are *derived from code*
(:mod:`aep.data.schema` + the actual frames), so they never drift from the
implementation. Re-run ``aep data dictionary`` / ``aep data quality`` to refresh.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from aep.config import REPO_ROOT
from aep.data.loader import TARGET_ENCODED, load_raw, process
from aep.data.schema import COLUMNS, LEAKAGE_COLUMNS, ColumnKind
from aep.data.source import BANK_MARKETING, DatasetSource, sha256_of, verify_checksum

DATA_DICTIONARY_PATH = REPO_ROOT / "data" / "kaggle" / "data_dictionary.md"
QUALITY_REPORT_PATH = REPO_ROOT / "reports" / "data-quality.md"
UNKNOWN_TOKEN = "unknown"
PDAYS_NOT_CONTACTED = 999


def write_data_dictionary(path: Path = DATA_DICTIONARY_PATH) -> Path:
    """Export the schema in :mod:`aep.data.schema` to a Markdown table."""
    lines = [
        "# Data dictionary — Bank Marketing (bank-additional-full)",
        "",
        "> Generated from `aep.data.schema` (`aep data dictionary`). Do not edit by hand.",
        "",
        f"Source: `{BANK_MARKETING.kaggle_ref}` (Kaggle) / UCI Bank Marketing. "
        f"Rows: {BANK_MARKETING.n_rows:,}. Separator: `{BANK_MARKETING.separator}`.",
        "",
        "| # | Column | Kind | Leakage | Description | Allowed values |",
        "|---|--------|------|---------|-------------|----------------|",
    ]
    for i, c in enumerate(COLUMNS, start=1):
        leak = "⛔ yes" if c.is_leakage else "—"
        cats = ", ".join(c.categories) if c.categories else ""
        lines.append(f"| {i} | `{c.name}` | {c.kind.value} | {leak} | {c.description} | {cats} |")

    lines += ["", "## Dropped columns (temporal / post-contact leakage)", ""]
    for name in LEAKAGE_COLUMNS:
        spec = next(c for c in COLUMNS if c.name == name)
        lines.append(f"- **`{name}`** — {spec.leakage_reason}")
    if not LEAKAGE_COLUMNS:
        lines.append("- (none)")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def compute_quality(df_raw: pd.DataFrame) -> dict:
    """Compute the data-quality facts used by the report."""
    df_proc = process(df_raw)
    n = len(df_raw)

    # 'unknown' is the dataset's missing-value token for categoricals.
    unknown_counts = {
        c.name: int((df_raw[c.name] == UNKNOWN_TOKEN).sum())
        for c in COLUMNS
        if c.kind is ColumnKind.CATEGORICAL and c.name in df_raw.columns
    }
    pos = int(df_proc[TARGET_ENCODED].sum())
    return {
        "n_rows": n,
        "n_cols_raw": df_raw.shape[1],
        "n_cols_processed": df_proc.shape[1],
        "n_duplicates": int(df_raw.duplicated().sum()),
        "nan_cells": int(df_raw.isna().sum().sum()),
        "target_positive": pos,
        "target_positive_rate": pos / n,
        "unknown_counts": {k: v for k, v in unknown_counts.items() if v > 0},
        "pdays_never_contacted": (
            int((df_raw["pdays"] == PDAYS_NOT_CONTACTED).sum()) if "pdays" in df_raw.columns else 0
        ),
        "numeric_describe": df_raw.describe().round(3),
    }


def write_quality_report(
    source: DatasetSource = BANK_MARKETING, path: Path = QUALITY_REPORT_PATH
) -> Path:
    """Generate `reports/data-quality.md` from the live data."""
    df_raw = load_raw(source, verify=False)
    q = compute_quality(df_raw)
    checksum_ok = verify_checksum(source)
    actual_sha = sha256_of(source.raw_path)

    lines = [
        "# Data quality report — Bank Marketing",
        "",
        f"_Generated on {date.today().isoformat()} by `aep data quality`._",
        "",
        "## 1. Provenance",
        "",
        f"- **Dataset:** {source.name}",
        f"- **Kaggle ref:** `{source.kaggle_ref}`",
        f"- **Upstream:** {source.upstream_url}",
        f"- **License:** {source.license_name} ({source.license_url})",
        f"- **Citation:** {source.citation}",
        f"- **Pinned SHA-256:** `{source.sha256}`",
        f"- **On-disk SHA-256:** `{actual_sha}` "
        f"({'✅ matches' if checksum_ok else '⚠️ MISMATCH'})",
        "",
        "## 2. Shape & integrity",
        "",
        f"- Rows: **{q['n_rows']:,}**",
        f"- Columns (raw): **{q['n_cols_raw']}** → (processed, leakage-free): "
        f"**{q['n_cols_processed']}**",
        f"- Exact duplicate rows: **{q['n_duplicates']:,}**",
        f"- True NaN cells: **{q['nan_cells']}** "
        "(categoricals use the literal token `unknown` instead of NaN)",
        "",
        "## 3. Target balance",
        "",
        f"- Positive class (`subscribed = 1`): **{q['target_positive']:,}** "
        f"of {q['n_rows']:,} = **{q['target_positive_rate']:.2%}**",
        "- The base is **imbalanced** (~1 in 9). Evaluation must not rely on raw "
        "accuracy; use balanced metrics and report per-segment exposure.",
        "",
        "## 4. Missingness (`unknown` token by column)",
        "",
        "| Column | # unknown | % |",
        "|--------|-----------|---|",
    ]
    for col, cnt in sorted(q["unknown_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{col}` | {cnt:,} | {cnt / q['n_rows']:.2%} |")
    lines += [
        "",
        f"- `pdays = 999` (never previously contacted): "
        f"**{q['pdays_never_contacted']:,}** rows "
        f"({q['pdays_never_contacted'] / q['n_rows']:.2%}). Treated as a sentinel, "
        "not a real day count.",
        "",
        "## 5. Temporal / post-contact leakage decision",
        "",
        "Dropped before any modelling (unavailable at decision time):",
        "",
    ]
    for name in LEAKAGE_COLUMNS:
        spec = next(c for c in COLUMNS if c.name == name)
        lines.append(f"- **`{name}`** — {spec.leakage_reason}")
    lines += [
        "",
        "Kept but flagged: `campaign` includes the current contact; `pdays`, "
        "`previous`, `poutcome` describe the *previous* campaign (not leakage). "
        "Socio-economic indicators (`emp.var.rate`, `cons.*`, `euribor3m`, "
        "`nr.employed`) are macro context known at decision time.",
        "",
        "## 6. Numeric summary (raw)",
        "",
        "```",
        q["numeric_describe"].to_string(),
        "```",
        "",
        "## 7. Limitations",
        "",
        "- Observational telemarketing data (2008-2010, one Portuguese bank); not "
        "representative of other institutions, products or periods.",
        "- No client identifiers, income, wealth or protected attributes are used.",
        "- Real outcomes seed the **synthetic** reward layer (Stage 2); they are "
        "not used as production labels.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
