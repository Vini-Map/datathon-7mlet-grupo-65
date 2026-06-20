"""Command-line interface for AEP (Typer).

The CLI is the single user-facing entry point and mirrors the Makefile targets.
Stage-specific subcommands are registered as the project grows; until a stage is
implemented they emit a clear "not yet implemented" notice instead of failing
silently, so the command surface is discoverable from day one.
"""

from __future__ import annotations

import typer
from rich.console import Console

from aep import __version__
from aep.config import get_settings
from aep.data.cli import app as data_app

app = typer.Typer(
    name="aep",
    help="Adaptive Experimentation Platform — bandit-based offer decisioning (Datathon 7MLET, Grupo 65).",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(data_app, name="data")
console = Console()


def _todo(stage: str, command: str) -> None:
    console.print(
        f"[yellow]>[/yellow] [bold]{command}[/bold] will be implemented in "
        f"[bold]{stage}[/bold]. The command surface is reserved so the workflow "
        f"is discoverable end-to-end.",
    )


@app.command()
def version() -> None:
    """Print the package version and active configuration summary."""
    settings = get_settings()
    console.print(f"[bold green]aep[/bold green] version [bold]{__version__}[/bold]")
    console.print(f"  llm_provider = {settings.llm_provider}")
    console.print(f"  random_seed  = {settings.random_seed}")
    console.print(f"  data_dir     = {settings.data_dir}")


@app.command()
def train() -> None:
    """(Stage 3) Train/simulate bandit policies and log runs to MLflow."""
    _todo("Stage 3", "aep train")


@app.command()
def eval() -> None:
    """(Stage 4) Run reproducible offline evaluation against the golden set."""
    _todo("Stage 4", "aep eval")


@app.command()
def serve() -> None:
    """(Stage 5) Start the FastAPI decision service."""
    _todo("Stage 5", "aep serve")


@app.command()
def demo() -> None:
    """(Stage 5) Run the end-to-end pipeline locally."""
    _todo("Stage 5", "aep demo")


if __name__ == "__main__":  # pragma: no cover
    app()
