"""CLI subcommands for the LLM+RAG assistant (Stage 5)."""

from __future__ import annotations

import typer
from rich.console import Console

from aep.assistant.assistant import Assistant

app = typer.Typer(help="Stage 5 — LLM+RAG assistant (summarize, retrieve policy).")
console = Console()


@app.command()
def summarize() -> None:
    """Summarize the latest experiment reports in plain language."""
    ans = Assistant().summarize_experiment()
    console.print(f"[bold]summary[/bold] ({ans.provider}):\n{ans.text}")


@app.command()
def policy(query: str, k: int = 3) -> None:
    """Retrieve relevant synthetic policy documents for a query."""
    for hit in Assistant().retrieve_policy(query, k=k):
        console.print(f"[bold]{hit.doc.doc_id}[/bold] ({hit.score:.3f}) — {hit.doc.title}")
        console.print(f"  {hit.doc.text}\n")
