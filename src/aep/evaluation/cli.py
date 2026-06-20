"""CLI subcommands for offline evaluation (Stage 4)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from aep.bandits.serving import ServingPolicy
from aep.evaluation.golden import build_golden_set, evaluate_golden
from aep.evaluation.report import write_evaluation_report

app = typer.Typer(help="Stage 4 — offline evaluation, golden set and fairness.")
console = Console()


@app.command()
def golden() -> None:
    """(Re)build the versioned golden set JSONL."""
    path = build_golden_set()
    console.print(f"[green]OK[/green] Golden set written: {path}")


@app.command()
def golden_run() -> None:
    """Evaluate the frozen serving policy against the golden set."""
    res = evaluate_golden(ServingPolicy().fit())
    table = Table(title=f"Golden set ({len(res)} cases)")
    for col in ("case_id", "category", "expected", "chosen", "passed"):
        table.add_column(col)
    for _, r in res.iterrows():
        table.add_row(
            r["case_id"],
            r["category"],
            r["expected_action"],
            r["chosen_action"],
            "[green]PASS[/green]" if r["passed"] else "[red]FAIL[/red]",
        )
    console.print(table)
    console.print(f"Pass rate: [bold]{res['passed'].mean():.1%}[/bold]")


@app.command()
def report() -> None:
    """Run the full offline evaluation and write reports/evaluation.md."""
    path = write_evaluation_report()
    console.print(f"[green]OK[/green] Evaluation report written: {path}")
