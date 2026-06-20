"""CLI subcommands for the data layer (Stage 1): build the processed base,
generate the data dictionary, and generate the data-quality report."""

from __future__ import annotations

import typer
from rich.console import Console

from aep.data.loader import RawDataNotFoundError, build_processed
from aep.data.reports import write_data_dictionary, write_quality_report
from aep.data.source import BANK_MARKETING, verify_checksum

app = typer.Typer(help="Stage 1 — Kaggle base loading, processing and reporting.")
console = Console()


@app.command()
def info() -> None:
    """Show provenance and whether the raw file is present and verified."""
    s = BANK_MARKETING
    present = s.raw_path.exists()
    console.print(f"[bold]{s.name}[/bold]")
    console.print(f"  kaggle_ref = {s.kaggle_ref}")
    console.print(f"  license    = {s.license_name}")
    console.print(f"  raw_path   = {s.raw_path} ({'present' if present else 'MISSING'})")
    if present:
        ok = verify_checksum(s)
        console.print(
            f"  checksum   = {'[green]matches pinned[/green]' if ok else '[red]MISMATCH[/red]'}"
        )


@app.command()
def build() -> None:
    """Load the raw CSV, drop leakage columns and write the processed Parquet."""
    try:
        df = build_processed()
    except RawDataNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]OK[/green] Processed base: {df.shape[0]:,} rows x {df.shape[1]} cols "
        "(leakage-free)."
    )


@app.command()
def dictionary() -> None:
    """Export the data dictionary to data/kaggle/data_dictionary.md."""
    path = write_data_dictionary()
    console.print(f"[green]OK[/green] Data dictionary written: {path}")


@app.command()
def quality() -> None:
    """Generate the data-quality report at reports/data-quality.md."""
    try:
        path = write_quality_report()
    except RawDataNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]OK[/green] Data-quality report written: {path}")


@app.command()
def all() -> None:  # noqa: A003 - intentional command name
    """Run build + dictionary + quality in sequence."""
    build()
    dictionary()
    quality()
