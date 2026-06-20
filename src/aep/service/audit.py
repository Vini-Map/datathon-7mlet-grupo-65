"""Auditable decision log (Stage 5).

Every decision is appended as one JSON line with a unique id, UTC timestamp, the
non-sensitive context, the selected arm, reason codes and the policy version.
This is the audit trail the banca inspects; it is the single source of truth for
"why was this offer shown to this context under which policy".
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from aep.config import REPO_ROOT

AUDIT_PATH = REPO_ROOT / "data" / "audit" / "decisions.jsonl"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def record_decision(context: dict, decision, path: Path = AUDIT_PATH) -> dict:
    """Append an audit record and return it (with audit_id + timestamp)."""
    record = {
        "audit_id": str(uuid.uuid4()),
        "timestamp": _now_iso(),
        "policy_version": decision.policy_version,
        "offer_id": decision.offer_id,
        "arm_index": decision.arm_index,
        "score": decision.score,
        "eligible_offers": decision.eligible_offers,
        "reason_codes": decision.reason_codes,
        "context": context,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_recent(n: int = 10, path: Path = AUDIT_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]
    return [json.loads(line) for line in lines[-n:]]
