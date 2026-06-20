"""CLI subcommands for the bandit policies (Stage 3)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from aep.bandits.report import write_bandit_report
from aep.bandits.runner import CONTEXTUAL, run_comparison

app = typer.Typer(help="Stage 3 — bandit baselines, policies, simulation and MLflow.")
console = Console()


@app.command()
def compare(
    n_steps: int = 20_000,
    seed: int = 123,
    delayed: bool = True,
    mlflow: bool = True,
) -> None:
    """Run all policies on one environment and print a metrics table."""
    results = run_comparison(n_steps=n_steps, seed=seed, delayed=delayed, log_mlflow=mlflow)
    table = Table(title=f"Bandit comparison ({n_steps:,} steps, delayed={delayed})")
    for col in ("policy", "ctx", "conversion", "cum_regret", "% optimal", "explore"):
        table.add_column(col, justify="right")
    for name, r in results.items():
        table.add_row(
            name,
            "yes" if name in CONTEXTUAL else "-",
            f"{r.conversion_rate:.2%}",
            f"{r.cum_regret:.1f}",
            f"{r.pct_optimal:.1%}",
            f"{r.exploration_entropy:.3f}",
        )
    console.print(table)


@app.command()
def report(n_steps: int = 20_000, seed: int = 123, mlflow: bool = True) -> None:
    """Run the full comparison + Nilos sweep and write reports/bandit-comparison.md."""
    path = write_bandit_report(n_steps=n_steps, seed=seed, log_mlflow=mlflow)
    console.print(f"[green]OK[/green] Bandit report written: {path}")
