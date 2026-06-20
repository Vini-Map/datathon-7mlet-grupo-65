"""Generate the Stage-7 MLOps monitoring report.

Runs the lifecycle demo on a throwaway registry and the drift monitors (stationary
baseline + a shocked window that fires the alerts), and writes
`reports/mlops-monitoring.md`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from aep.config import REPO_ROOT
from aep.mlops.drift import PSI_ALERT, PSI_WARN, monitor_drift
from aep.mlops.lifecycle import run_lifecycle_demo
from aep.mlops.registry import PolicyRegistry

REPORT_PATH = REPO_ROOT / "reports" / "mlops-monitoring.md"
_DEMO_REGISTRY = REPO_ROOT / "models" / "demo_policy_registry.json"


def write_mlops_report(path: Path = REPORT_PATH) -> Path:
    # Fresh registry so the demo is reproducible.
    _DEMO_REGISTRY.unlink(missing_ok=True)
    registry = PolicyRegistry(path=_DEMO_REGISTRY)
    steps = run_lifecycle_demo(registry)

    baseline = monitor_drift()
    shocked = monitor_drift(shock="both")

    prod = registry.production()
    lines = [
        "# MLOps monitoring report (Stage 7)",
        "",
        f"_Generated on {date.today().isoformat()} by `aep mlops report`._",
        "",
        "## 1. Lifecycle demo — experiment to controlled production",
        "",
        "A new offer-policy hypothesis (`linucb-v2`, more exploration) goes from "
        "experiment to production through the automated gate **and** a human "
        "approval, then is rolled back — exactly the controlled path required.",
        "",
        "| Step | Detail |",
        "|------|--------|",
    ]
    for s in steps:
        lines.append(f"| {s.action} | {s.detail} |")

    lines += [
        "",
        f"Final production policy after the demo (post-rollback): "
        f"**`{prod.version if prod else 'none'}`**.",
        "",
        "Registry stages and approvals:",
        "",
        "| Version | Stage | Approved by | Conversion | Golden | Fairness |",
        "|---------|-------|-------------|-----------:|-------:|---------:|",
    ]
    for v in registry.list():
        lines.append(
            f"| `{v.version}` | {v.stage} | {v.approver or '—'} | "
            f"{v.metrics.get('conversion', 0):.2%} | "
            f"{v.metrics.get('golden_pass_rate', 0):.0%} | "
            f"{v.metrics.get('fairness_disparity', 0):.2f} |"
        )

    lines += [
        "",
        "## 2. Drift monitoring (PSI per context feature)",
        "",
        f"Thresholds: PSI >= {PSI_WARN} warn, PSI >= {PSI_ALERT} alert. The natural "
        "stream is stationary, so the baseline shows no drift; the shocked window "
        "(age/euribor shift + halved reward) confirms the detectors fire.",
        "",
        "| Feature | PSI (baseline) | PSI (shocked) |",
        "|---------|---------------:|--------------:|",
    ]
    for feat in baseline.feature_psi:
        lines.append(
            f"| `{feat}` | {baseline.feature_psi[feat]:.3f} | " f"{shocked.feature_psi[feat]:.3f} |"
        )

    lines += [
        "",
        f"- Baseline: max PSI **{baseline.max_psi:.3f}** "
        f"(data alert: {'YES' if baseline.data_alert else 'no'}).",
        f"- Shocked: max PSI **{shocked.max_psi:.3f}** "
        f"(data alert: {'YES' if shocked.data_alert else 'no'}).",
        "",
        "## 3. Reward drift",
        "",
        "| Window | Reference conv. | Current conv. | Rel. change | Alert |",
        "|--------|----------------:|--------------:|------------:|:-----:|",
        f"| baseline | {baseline.reward_reference:.2%} | {baseline.reward_current:.2%} | "
        f"{baseline.reward_relative_change:+.1%} | "
        f"{'YES' if baseline.reward_alert else 'no'} |",
        f"| shocked | {shocked.reward_reference:.2%} | {shocked.reward_current:.2%} | "
        f"{shocked.reward_relative_change:+.1%} | "
        f"{'YES' if shocked.reward_alert else 'no'} |",
        "",
        "## 4. How this maps to the lifecycle",
        "",
        "- A drift alert (data or reward) **triggers a retraining run**; the candidate "
        "is registered in staging and must clear the promotion gate.",
        "- The promotion gate enforces hard guardrails (adversarial suitability), "
        "golden-set quality and no conversion/fairness regression.",
        "- A **human approver** signs off before production; **rollback** is one call "
        "back to the previous archived version.",
        "- Experiments and policy versions are tracked in MLflow (Stage 3 logs runs; "
        "the Azure target uses the Azure ML Model Registry stages).",
        "",
        "See [`docs/mlops-plan.md`](../docs/mlops-plan.md) for the full policy.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
