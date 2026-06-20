"""Generate the Stage-3 bandit comparison report and regret figure.

Runs the full comparison (logging to MLflow), the Nilos-UCB confidence sweep, and
writes `reports/bandit-comparison.md` plus a cumulative-regret figure used by the
technical report.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aep.bandits.environment import build_environment  # noqa: E402
from aep.bandits.runner import CONTEXTUAL, run_comparison, sweep_nilos  # noqa: E402
from aep.config import REPO_ROOT  # noqa: E402

REPORT_PATH = REPO_ROOT / "reports" / "bandit-comparison.md"
FIGURE_PATH = REPO_ROOT / "reports" / "figures" / "bandit_regret.png"


def _plot_regret(results, path: Path = FIGURE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        ax.plot(r.regret_curve, label=name, linewidth=1.5)
    ax.set_xlabel("decision step")
    ax.set_ylabel("cumulative expected regret")
    ax.set_title("Cumulative regret by policy (lower is better)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_bandit_report(n_steps: int = 20_000, seed: int = 123, log_mlflow: bool = True) -> Path:
    env = build_environment(n_steps=n_steps)
    results = run_comparison(n_steps=n_steps, seed=seed, log_mlflow=log_mlflow, env=env)
    sweep = sweep_nilos(n_steps=n_steps, seed=seed, env=env)
    _plot_regret(results)

    baseline = results["greedy_baseline"]
    best = min(results.values(), key=lambda r: r.cum_regret)

    lines = [
        "# Bandit policy comparison (Stage 3)",
        "",
        f"_Generated on {date.today().isoformat()} by `aep bandits report`. "
        f"{n_steps:,} steps, delayed feedback, seed={seed}. Logged to MLflow._",
        "",
        "## 1. Headline",
        "",
        f"- Lowest cumulative regret: **{best.policy}** "
        f"(regret {best.cum_regret:.1f}, conversion {best.conversion_rate:.2%}, "
        f"{best.pct_optimal:.1%} optimal actions).",
        f"- Deterministic baseline (`greedy_baseline`): regret {baseline.cum_regret:.1f}, "
        f"conversion {baseline.conversion_rate:.2%}, {baseline.pct_optimal:.1%} optimal.",
        f"- The contextual policies ({', '.join(sorted(CONTEXTUAL))}) use the client "
        "context to route each segment to its best arm, which a context-free policy "
        "cannot do — see `% optimal`.",
        "",
        "## 2. Metrics (all policies)",
        "",
        "| Policy | Contextual | Conversion | Cum. reward | Cum. regret | % optimal | Exploration (entropy) |",
        "|--------|:---------:|-----------:|------------:|------------:|----------:|----------------------:|",
    ]
    order = ["random", "greedy_baseline", "thompson", "ucb1", "nilos_ucb", "linucb", "neural"]
    for name in order:
        r = results[name]
        ctx = "yes" if name in CONTEXTUAL else "—"
        lines.append(
            f"| `{name}` | {ctx} | {r.conversion_rate:.2%} | {r.cum_reward:.0f} | "
            f"{r.cum_regret:.1f} | {r.pct_optimal:.1%} | {r.exploration_entropy:.3f} |"
        )

    lines += [
        "",
        "Exploration entropy is the normalized entropy of the arm-selection "
        "distribution (0 = always one arm, 1 = uniform). `random` sits near 1; the "
        "baseline collapses to one arm; adaptive policies sit in between.",
        "",
        "![Cumulative regret by policy](figures/bandit_regret.png)",
        "",
        "## 3. Nilos-UCB confidence x exploration x conversion trade-off",
        "",
        "Sweeping the confidence coefficient `c` (index = mean + c·sqrt(ln t / n)):",
        "",
        "| c | Conversion | Cum. regret | % optimal | Exploration (entropy) |",
        "|---|-----------:|------------:|----------:|----------------------:|",
    ]
    for c, r in sweep.items():
        lines.append(
            f"| {c} | {r.conversion_rate:.2%} | {r.cum_regret:.1f} | "
            f"{r.pct_optimal:.1%} | {r.exploration_entropy:.3f} |"
        )
    lines += [
        "",
        "Small `c` exploits early (good short-term conversion, but risks locking "
        "onto a sub-optimal arm); large `c` explores more (higher entropy, better "
        "long-run optimality). This is the confidence/exploration knob the banca "
        "asks us to analyze for the UCB family.",
        "",
        "## 4. Cold-start handling",
        "",
        "- **Thompson:** uninformative Beta(1,1) priors — unplayed arms are sampled "
        "fairly from the prior, so early picks are exploratory by construction.",
        "- **UCB1 / Nilos-UCB:** unplayed arms get an infinite index, forcing one "
        "pull of every eligible arm before exploitation.",
        "- **Greedy baseline:** explicit one-pull-per-arm warm-up, then commit.",
        "- **LinUCB:** ridge prior (A = I) gives a large initial confidence bonus "
        "that shrinks as evidence accrues.",
        "- **Neural:** epsilon starts near 1 and the net is untrained until the "
        "replay buffer fills, so it behaves like `random` during cold-start.",
        "",
        "## 5. Delayed-reward handling",
        "",
        "The simulator delivers each reward `1 + Poisson(lambda_product)` steps "
        "after the decision (a non-trivial delay for slow-settling products like "
        "investment/premium). Policies therefore act on **stale statistics** and "
        "must keep exploring until feedback arrives — this is why purely greedy "
        "exploitation is penalized and uncertainty-aware policies (Thompson, UCB, "
        "LinUCB) are more robust. Rewards still pending after the last step are "
        "flushed so no feedback is lost.",
        "",
        "## 6. Reproducibility",
        "",
        "All policies see the **same** environment stream and common-random-number "
        "reward draws (variance reduction). Seeds are fixed; every policy is a "
        "separate MLflow run under the configured experiment with its "
        "hyper-parameters and metrics.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT_PATH
