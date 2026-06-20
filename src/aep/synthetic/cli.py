"""CLI subcommands for the synthetic enrichment layer (Stage 2)."""

from __future__ import annotations

import typer
from rich.console import Console

from aep.synthetic.generate import generate as run_generate
from aep.synthetic.report import write_data_generation_report, write_schema_readme

app = typer.Typer(help="Stage 2 — synthetic offer catalog, events and delayed rewards.")
console = Console()


@app.command()
def generate() -> None:
    """Generate offer_catalog, offer_events and delayed_rewards (seeded)."""
    data = run_generate()
    for name, df in data.items():
        console.print(f"[green]OK[/green] {name}: {df.shape[0]:,} rows x {df.shape[1]} cols")


@app.command()
def report() -> None:
    """Write the schema README and reports/data-generation.md."""
    p1 = write_schema_readme()
    p2 = write_data_generation_report()
    console.print(f"[green]OK[/green] Schema README: {p1}")
    console.print(f"[green]OK[/green] Data-generation report: {p2}")


@app.command()
def all() -> None:  # noqa: A003
    """Generate data then write documentation."""
    generate()
    report()
