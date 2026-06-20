# Technical report — Adaptive Experimentation Platform (AEP)

**Datathon 7MLET / POSTECH — Fase 05 — Grupo 65**

> Academic/demonstration project. It does **not** simulate a real bank and is
> **not** validated for production-regulated use. Every experimentation artifact
> is synthetic, seeded and reproducible.

---

## 1. Problem & framing

A digital bank must decide, per channel and per eligible client, **which offer,
message or next step** to present. Fixed rules and long A/B tests are slow to
adapt and waste exposure. We frame this as a **multi-armed bandit**: each offer
is an *arm*; the goal is to maximize conversion (reward) while minimizing
*regret*, balancing **exploration** and **exploitation**, and coping with
**cold-start** and **delayed rewards**.

The platform is end-to-end: a reproducible data pipeline, bandit policies with
MLflow tracking, offline evaluation with a golden set, a FastAPI/CLI service with
an LLM+RAG assistant, an Azure target architecture, an MLOps lifecycle, and
governance artifacts. A single command (`aep demo` / `make demo`) reproduces it.

## 2. Data — factual base & leakage

- **Base:** UCI/Kaggle *Bank Marketing* (`bank-additional-full`): 41,188 rows,
  20 inputs + binary target `y` (term-deposit subscription), CC BY 4.0. Source,
  version and a **SHA-256 checksum** are pinned in code and verified on load.
- **Imbalance:** 11.27% positive — evaluation avoids raw accuracy.
- **Temporal-leakage decision:** `duration` (call length, known only *after*
  contact; ~perfectly encodes the outcome) is **dropped**. `campaign`, `pdays`,
  `previous`, `poutcome` and macro indicators are kept and documented. Details in
  `reports/data-quality.md` and `data/kaggle/data_dictionary.md`.

## 3. Synthetic enrichment

The base has no offers or bandit feedback, so we synthesize a commercial layer,
**physically separate** and fully seeded (master seed 42):

- **`offer_catalog`** — 8 arms (product × channel × message tone) with explicit
  **eligibility/suitability** rules (e.g. no credit/loan/investment for defaulting
  clients; Premium Advisory restricted to white-collar, age ≥ 30).
- **`offer_events`** — 20,000 logged impressions over a 90-day horizon, with the
  logged action and an **exact propensity** (uniform over eligible ⇒ 1/n_eligible),
  enabling unbiased off-policy evaluation, plus an immediate click reward.
- **`delayed_rewards`** — conversion realized after `1 + Poisson(λ_product)` days;
  non-converters censored at 14 days.
- **Grounded oracle reward model** — `P(reward) = P(click)·P(conv|click)` with
  context×offer affinities; the **optimal arm depends on the segment** (e.g. O1
  for mid non-white-collar, O8 for white-collar, O7 for seniors), which is the
  empirical justification for a contextual policy. See `reports/data-generation.md`.

## 4. Bandit modeling

Implemented from scratch (`aep.bandits`):

- **Deterministic baseline** — warm-up one pull per arm, then exploit the best
  historical arm.
- **Thompson Sampling** — Beta-Bernoulli, uninformative Beta(1,1) priors.
- **UCB1** and **Nilos-UCB** — confidence-parameterized UCB
  (`index = mean + c·sqrt(ln t / n)`); sweeping `c` traces the
  confidence × exploration × conversion trade-off.
- **LinUCB** — disjoint linear contextual bandit (ridge prior, UCB bonus).
- **Neural bandit (PyTorch)** — an MLP reward estimator with decaying
  epsilon-greedy exploration and a replay buffer.

**Cold-start:** Thompson priors, UCB infinite index for unplayed arms, LinUCB
ridge prior, neural high-epsilon warm-up. **Delayed rewards:** the simulator
delivers each reward several steps later, so policies act on stale statistics —
penalizing pure exploitation and rewarding uncertainty-aware policies. All runs
are logged to **MLflow** (params, metrics) under the `aep-bandits` experiment.

## 5. Quantitative comparison (baseline × adaptive)

20,000 steps, delayed feedback, common-random-number rewards
(`reports/bandit-comparison.md`):

| Policy | Contextual | Conversion | Cum. regret | % optimal | Exploration |
|--------|:---------:|-----------:|------------:|----------:|------------:|
| random | — | 6.17% | 1676 | 13.9% | 0.980 |
| greedy_baseline | — | 8.61% | 1166 | 26.9% | 0.009 |
| thompson | — | 10.61% | 739 | 41.8% | 0.636 |
| ucb1 | — | 9.45% | 983 | 34.5% | 0.899 |
| nilos_ucb | — | 9.74% | 928 | 35.7% | 0.864 |
| **linucb** | **yes** | **13.12%** | **240** | **69.1%** | 0.913 |
| neural | yes | 11.71% | 531 | 51.5% | 0.904 |

**LinUCB** dominates: lowest regret and highest conversion/optimality, because
the best arm depends on context. The deterministic baseline collapses onto a
single arm (entropy 0.009) and cannot adapt. Thompson is the strongest
context-free policy.

## 6. Offline evaluation & golden set

`reports/evaluation.md`, frozen `linucb-v1`:

- **Off-policy value:** SNIPS **13.70%**, IPS 13.39% — within ~0.4 pp of the
  **oracle-true** value (13.81%), validating the OPE method; **+133% lift** over
  the uniform logging policy (5.93%); optimal upper bound 14.47%.
- **Golden set:** 22 versioned cases (typical / per-segment / edge / adversarial)
  with explicit pass criteria. **Pass rate 90.9%**; **all adversarial suitability
  guardrails pass** (the policy never selects an ineligible/forbidden offer). The
  two failures honestly flag segments where the linear policy under-routes.
- **Sensitivity:** stable across serving `alpha`.
- **Fairness:** exposure value disparity 0.29 (min/max across segments) — flagged
  as a limitation and monitored.

## 7. Service & assistant

- **FastAPI + CLI** (`aep decide`, `aep serve`): context → decision with reason
  codes, selected arm and policy version; every decision is written to an
  **auditable JSONL log**. Documented Pydantic I/O contract and error handling
  (e.g. 503 when the base is missing, 422 on invalid input).
- **LLM+RAG assistant** behind a pluggable `LLMProvider` (offline MockProvider by
  default; Anthropic optional) over a TF-IDF retriever on synthetic policy docs:
  explains decisions (grounded, with citations), retrieves policy, summarizes
  experiments.
- **One command:** `aep demo` runs data → synthetic → bandits → evaluation → an
  explained decision end-to-end.

## 8. Azure target architecture

100% Azure mapping (`docs/architecture-azure.md`): **Container Apps** (serving),
**API Management** (gateway), **Azure ML** (training + MLflow registry stages),
**Azure OpenAI + AI Search** (assistant/RAG), **ADLS Gen2 + PostgreSQL** (data +
audit), **Azure Monitor/App Insights** (observability), **Entra ID + Managed
Identity** (identity), **Key Vault** (secrets), **Purview** (governance). Includes
a Mermaid diagram, deploy plan, qualitative FinOps levers (scale-to-zero,
scheduled retrain, LLM/RAG caching) and trade-offs (ACA vs AKS, PostgreSQL vs
Cosmos, AI Search vs self-hosted FAISS).

## 9. MLOps lifecycle

`docs/mlops-plan.md`, `reports/mlops-monitoring.md`:

- **Policy registry** with `staging → production → archived` stages, an explicit
  **human approval gate** (promotion blocked until approved), and **one-call
  rollback**.
- **Promotion gate:** a hard adversarial-guardrail check (never waived) plus
  golden-set, fairness-floor and no-regression checks.
- **Drift monitoring:** PSI per context feature + reward-rate change; alerts
  trigger retraining. A shocked window confirms the detectors fire (max PSI 10.1,
  reward −47.6%).
- Demonstrated: a new hypothesis (`linucb-v2`) goes experiment → staging → gate →
  approval → production → rollback.

## 10. Limitations, risks & hypotheses

- **Synthetic rewards:** absolute numbers are not real demand; only relative
  comparisons hold. The real `y` only *grounds* the oracle.
- **Single base/period; linear-model bias** (mis-routes some segments);
  **exposure disparity** 0.29.
- **Risks** (see `system-card.md`): reward hacking, context manipulation,
  assistant abuse, suitability violation, silent drift — each with a mitigation.
- **Encoded hypotheses:** conversion is heterogeneous by segment; channels differ
  in engagement; premium/investment suit older white-collar segments; rewards are
  delayed/censored.

## 11. Future work

- Non-linear contextual policies (neural-linear / Thompson over the neural model)
  to fix the segments LinUCB under-routes.
- Doubly-robust off-policy estimators; richer fairness constraints in the gate.
- Real propensity logging and a proper online/offline feedback loop on Azure.
- Counterfactual exposure budgets and diversity constraints.

## 12. References

- Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI ML
  Repository. https://doi.org/10.24432/C5K306 (CC BY 4.0).
- Li, Chu, Langford & Schapire (2010). *A Contextual-Bandit Approach to
  Personalized News Article Recommendation* (LinUCB).
- Auer, Cesa-Bianchi & Fischer (2002). *Finite-time Analysis of the Multiarmed
  Bandit Problem* (UCB1).
- Thompson (1933). *On the Likelihood that One Unknown Probability Exceeds
  Another...* (Thompson Sampling).
- Swaminathan & Joachims (2015). *The Self-Normalized Estimator for Counterfactual
  Learning* (SNIPS).
- Project artifacts: `reports/` (data-quality, data-generation, bandit-comparison,
  evaluation, mlops-monitoring) and `docs/` (architecture-azure, model-card,
  system-card, lgpd-plan, mlops-plan).
