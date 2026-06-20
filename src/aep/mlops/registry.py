"""Policy registry with stages, an approval gate and rollback (Stage 7).

A lightweight JSON-backed registry that versions policies and moves them through
``staging -> production -> archived`` under an explicit **human approval gate**:
a candidate cannot be promoted to production until someone approves it. Rollback
re-promotes the previously-archived production version. This mirrors the Azure ML
Model Registry stages in the target architecture, but runs locally and is fully
testable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from aep.config import REPO_ROOT

REGISTRY_PATH = REPO_ROOT / "models" / "policy_registry.json"

STAGE_STAGING = "staging"
STAGE_PRODUCTION = "production"
STAGE_ARCHIVED = "archived"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class PolicyVersion:
    version: str
    stage: str
    metrics: dict
    params: dict
    approved: bool = False
    approver: str | None = None
    approval_notes: str | None = None
    created_at: str = field(default_factory=_now)
    promoted_at: str | None = None


class ApprovalError(RuntimeError):
    """Raised when promotion is attempted without passing the approval gate."""


class PolicyRegistry:
    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        self.path = path
        self._versions: dict[str, PolicyVersion] = {}
        self._load()

    # --- persistence --------------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._versions = {v["version"]: PolicyVersion(**v) for v in data["versions"]}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"versions": [asdict(v) for v in self._versions.values()]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- registration / approval / promotion --------------------------------

    def register(self, version: str, metrics: dict, params: dict) -> PolicyVersion:
        """Register a new candidate in ``staging``."""
        if version in self._versions:
            raise ValueError(f"version {version} already exists")
        pv = PolicyVersion(version=version, stage=STAGE_STAGING, metrics=metrics, params=params)
        self._versions[version] = pv
        self._save()
        return pv

    def approve(self, version: str, approver: str, notes: str = "") -> PolicyVersion:
        """Record a human approval (the approval gate)."""
        pv = self._require(version)
        pv.approved = True
        pv.approver = approver
        pv.approval_notes = notes
        self._save()
        return pv

    def promote(self, version: str) -> PolicyVersion:
        """Promote an approved candidate to production; archive the incumbent."""
        pv = self._require(version)
        if not pv.approved:
            raise ApprovalError(
                f"version {version} is not approved; an approver must sign off first."
            )
        for other in self._versions.values():
            if other.stage == STAGE_PRODUCTION:
                other.stage = STAGE_ARCHIVED
        pv.stage = STAGE_PRODUCTION
        pv.promoted_at = _now()
        self._save()
        return pv

    def rollback(self) -> PolicyVersion:
        """Roll back to the most recently archived production version."""
        archived = [v for v in self._versions.values() if v.stage == STAGE_ARCHIVED]
        if not archived:
            raise RuntimeError("no archived version to roll back to")
        target = max(archived, key=lambda v: v.promoted_at or "")
        for other in self._versions.values():
            if other.stage == STAGE_PRODUCTION:
                other.stage = STAGE_ARCHIVED
        target.stage = STAGE_PRODUCTION
        target.promoted_at = _now()
        self._save()
        return target

    # --- queries ------------------------------------------------------------

    def production(self) -> PolicyVersion | None:
        for v in self._versions.values():
            if v.stage == STAGE_PRODUCTION:
                return v
        return None

    def list(self) -> list[PolicyVersion]:
        return sorted(self._versions.values(), key=lambda v: v.created_at)

    def _require(self, version: str) -> PolicyVersion:
        if version not in self._versions:
            raise KeyError(f"unknown version {version}")
        return self._versions[version]
