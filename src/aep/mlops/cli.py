"""CLI subcommands for the MLOps lifecycle (Stage 7)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from aep.mlops.drift import monitor_drift
from aep.mlops.registry import ApprovalError, PolicyRegistry
from aep.mlops.report import write_mlops_report

app = typer.Typer(help="Stage 7 — policy registry, promotion, rollback and drift.")
console = Console()


@app.command()
def status() -> None:
    """Show the registry: versions, stages and the current production policy."""
    reg = PolicyRegistry()
    prod = reg.production()
    table = Table(title="Policy registry")
    for col in ("version", "stage", "approved", "conversion", "golden"):
        table.add_column(col)
    for v in reg.list():
        table.add_row(
            v.version,
            v.stage,
            "yes" if v.approved else "no",
            f"{v.metrics.get('conversion', 0):.2%}",
            f"{v.metrics.get('golden_pass_rate', 0):.0%}",
        )
    console.print(table)
    console.print(f"production: [bold]{prod.version if prod else 'none'}[/bold]")


@app.command()
def approve(version: str, approver: str, notes: str = "") -> None:
    """Record a human approval for a candidate (the approval gate)."""
    PolicyRegistry().approve(version, approver=approver, notes=notes)
    console.print(f"[green]OK[/green] {version} approved by {approver}")


@app.command()
def promote(version: str) -> None:
    """Promote an approved candidate to production."""
    try:
        PolicyRegistry().promote(version)
    except ApprovalError as exc:
        console.print(f"[red]blocked[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]OK[/green] {version} promoted to production")


@app.command()
def rollback() -> None:
    """Roll back to the previous archived production version."""
    target = PolicyRegistry().rollback()
    console.print(f"[green]OK[/green] rolled back to {target.version}")


@app.command()
def drift(shock: str = "") -> None:
    """Run the drift monitors (optionally with a shock: data|reward|both)."""
    rep = monitor_drift(shock=shock or None)
    console.print(f"max PSI: {rep.max_psi:.3f} (data alert: {rep.data_alert})")
    console.print(
        f"reward: {rep.reward_reference:.2%} -> {rep.reward_current:.2%} "
        f"({rep.reward_relative_change:+.1%}, alert: {rep.reward_alert})"
    )


@app.command()
def report() -> None:
    """Run the lifecycle demo + drift monitors and write the report."""
    path = write_mlops_report()
    console.print(f"[green]OK[/green] MLOps report written: {path}")
