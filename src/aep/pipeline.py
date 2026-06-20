"""End-to-end pipeline (Stage 5).

`run_pipeline` reproduces the whole project locally in one call: process the
base, generate the synthetic layer, run the bandit comparison (MLflow), run the
offline evaluation, then make and explain one example decision. This backs the
single `aep demo` / `make demo` command.
"""

from __future__ import annotations

import pandas as pd
from rich.console import Console

from aep.bandits.report import write_bandit_report
from aep.bandits.serving import ServingPolicy
from aep.data.loader import build_processed
from aep.data.reports import write_data_dictionary, write_quality_report
from aep.evaluation.report import write_evaluation_report
from aep.synthetic.generate import generate
from aep.synthetic.report import write_data_generation_report, write_schema_readme

console = Console()

_EXAMPLE_CONTEXT = {
    "age": 45,
    "job": "management",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
    "subscribed": 0,
}


def run_pipeline(n_steps: int = 20_000, log_mlflow: bool = True) -> dict:
    """Run every stage end-to-end; return artifact paths and the example decision."""
    console.rule("[bold]1/5 Data layer")
    build_processed()
    write_data_dictionary()
    write_quality_report()

    console.rule("[bold]2/5 Synthetic enrichment")
    generate()
    write_schema_readme()
    write_data_generation_report()

    console.rule("[bold]3/5 Bandit comparison (MLflow)")
    bandit_report = write_bandit_report(n_steps=n_steps, log_mlflow=log_mlflow)

    console.rule("[bold]4/5 Offline evaluation")
    eval_report = write_evaluation_report()

    console.rule("[bold]5/5 Example decision + explanation")
    from aep.assistant.assistant import Assistant

    policy = ServingPolicy().fit()
    decision = policy.decide_row(pd.DataFrame([_EXAMPLE_CONTEXT]))
    explanation = Assistant().explain_decision(decision, _EXAMPLE_CONTEXT)

    console.print(
        f"  selected offer : [bold]{decision.offer_id}[/bold] "
        f"({decision.scores[decision.offer_id]:.3f})"
    )
    console.print(f"  policy version : {decision.policy_version}")
    console.print(f"  reason codes   : {decision.reason_codes}")
    console.print(f"  assistant      : {explanation.provider} (cites {explanation.citations})")

    return {
        "bandit_report": str(bandit_report),
        "eval_report": str(eval_report),
        "decision": decision.offer_id,
        "policy_version": decision.policy_version,
    }
