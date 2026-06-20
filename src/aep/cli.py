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
from aep.assistant.cli import app as assistant_app
from aep.bandits.cli import app as bandits_app
from aep.config import get_settings
from aep.data.cli import app as data_app
from aep.evaluation.cli import app as eval_app
from aep.mlops.cli import app as mlops_app
from aep.synthetic.cli import app as synth_app

app = typer.Typer(
    name="aep",
    help="Adaptive Experimentation Platform — bandit-based offer decisioning (Datathon 7MLET, Grupo 65).",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(data_app, name="data")
app.add_typer(synth_app, name="synth")
app.add_typer(bandits_app, name="bandits")
app.add_typer(eval_app, name="eval")
app.add_typer(assistant_app, name="assistant")
app.add_typer(mlops_app, name="mlops")
console = Console()


@app.command()
def version() -> None:
    """Print the package version and active configuration summary."""
    settings = get_settings()
    console.print(f"[bold green]aep[/bold green] version [bold]{__version__}[/bold]")
    console.print(f"  llm_provider = {settings.llm_provider}")
    console.print(f"  random_seed  = {settings.random_seed}")
    console.print(f"  data_dir     = {settings.data_dir}")


@app.command()
def train(n_steps: int = 20_000, seed: int = 123) -> None:
    """(Stage 3) Simulate all bandit policies, log to MLflow, write the report."""
    from aep.bandits.report import write_bandit_report

    path = write_bandit_report(n_steps=n_steps, seed=seed, log_mlflow=True)
    console.print(f"[green]OK[/green] Bandit comparison report: {path}")


@app.command()
def serve(host: str | None = None, port: int | None = None) -> None:
    """(Stage 5) Start the FastAPI decision + assistant service."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "aep.service.app:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
    )


@app.command()
def decide(
    age: int = 45,
    job: str = "management",
    default: str = "no",
    contact: str = "cellular",
    explain: bool = False,
) -> None:
    """(Stage 5) Make one offer decision from the command line."""
    import pandas as pd

    from aep.bandits.serving import ServingPolicy
    from aep.service import audit
    from aep.service.schemas import ClientContext

    ctx = ClientContext(age=age, job=job, default=default, contact=contact)
    features = ctx.to_features()
    decision = ServingPolicy().fit().decide_row(pd.DataFrame([features]))
    record = audit.record_decision(features, decision)
    console.print(
        f"[bold green]offer[/bold green]   : {decision.offer_id} " f"(score {decision.score:.3f})"
    )
    console.print(f"policy  : {decision.policy_version}")
    console.print(f"reasons : {decision.reason_codes}")
    console.print(f"audit_id: {record['audit_id']}")
    if explain:
        from aep.assistant.assistant import Assistant

        ans = Assistant().explain_decision(decision, features)
        console.print(f"\n[bold]explanation[/bold] ({ans.provider}, cites {ans.citations}):")
        console.print(ans.text)


@app.command()
def demo(n_steps: int = 20_000, mlflow: bool = True) -> None:
    """(Stage 5) Run the end-to-end pipeline locally (data -> train -> eval -> decision)."""
    from aep.pipeline import run_pipeline

    out = run_pipeline(n_steps=n_steps, log_mlflow=mlflow)
    console.print(
        f"\n[green]OK[/green] Pipeline complete. Example decision: "
        f"[bold]{out['decision']}[/bold] ({out['policy_version']})."
    )


if __name__ == "__main__":  # pragma: no cover
    app()
