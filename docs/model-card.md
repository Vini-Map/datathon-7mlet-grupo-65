# Model Card — AEP offer-decisioning policy

> Academic/demonstration artifact for the 7MLET Datathon (Grupo 65). **Not** a
> production model and **not** validated for real regulated use.

## Model details

- **Name / version:** `linucb-v1` (frozen LinUCB serving policy).
- **Type:** Disjoint linear contextual bandit (LinUCB), frozen for serving by
  scoring the learned mean reward per arm (no exploration bonus at decision time).
- **Family evaluated:** deterministic greedy baseline, Thompson Sampling
  (Beta-Bernoulli), UCB1, Nilos-UCB, LinUCB, and a PyTorch neural bandit.
- **Inputs:** 11 non-sensitive context features (age, previous contacts, three
  macro indicators, and binary segment indicators) derived from the processed
  Bank Marketing base. **No** identifiers, income, wealth, gender or race.
- **Outputs:** one eligible offer (arm) from an 8-offer catalog, with per-arm
  scores, reason codes and the policy version.
- **Owners:** Datathon 7MLET — Grupo 65.

## Training & evaluation data

- **Factual base:** UCI/Kaggle *Bank Marketing* (`bank-additional-full`, 41,188
  rows, CC BY 4.0). The post-contact leakage column `duration` is dropped.
- **Experimentation layer:** fully **synthetic**, seeded (master seed 42): an
  8-arm offer catalog, 20,000 logged impressions with exact propensities, a
  grounded oracle reward model, and delayed/censored rewards.
- **Evaluation:** 22-case versioned golden set + off-policy estimation (IPS/SNIPS)
  on the logged data. See `reports/evaluation.md`.

## Metrics (frozen `linucb-v1`)

| Metric | Value |
|--------|------:|
| Off-policy value — SNIPS | 13.70% |
| Off-policy value — IPS | 13.39% |
| Oracle-true value | 13.81% |
| Logging (uniform) value | 5.93% |
| Optimal (oracle) upper bound | 14.47% |
| Relative lift vs uniform logging | **+133%** |
| Golden-set pass rate | 90.9% (adversarial 100%) |
| Online cumulative regret (20k steps) | 239.8 (lowest of all policies) |
| % optimal actions (online) | 69.1% |
| Exposure value disparity (min/max segment) | 0.29 |

IPS/SNIPS (estimated from logged data only) land within ~0.4 pp of the
oracle-true value, validating the off-policy methodology.

## Intended use

- Demonstrate **adaptive experimentation** (bandits) for offer/channel/message
  selection in a **synthetic** regulated-finance scenario.
- Educational reference for problem framing, evaluation, MLOps and governance.

## Out-of-scope use

- Any decision affecting **real** customers, money, credit or eligibility.
- Use on real personal data, or with protected/sensitive attributes.
- Treating the synthetic oracle's absolute rates as real-world demand.
- Deployment without human-in-the-loop review and the Stage-7 approval gate.

## Fairness analysis

- Exposure and **delivered value** are measured across synthetic segments
  (age band × occupation group). Value disparity is **0.29** (min/max), i.e. the
  policy delivers materially more expected conversion to some segments — this is
  monitored and must be reviewed before any rollout.
- `segment` is a synthetic operational construct, **not** a protected attribute;
  this is an exposure check, not a legal fairness audit.

## Known biases & limitations

- **Synthetic rewards:** all values come from a modeled oracle; only relative
  comparisons are meaningful.
- **Linear-model bias:** the frozen LinUCB mis-routes a few segments (2 of 22
  golden cases), over-selecting broadly-appealing offers; do not use for those
  segments without monitoring.
- **Single base, single period:** Bank Marketing is one Portuguese bank,
  2008–2010; not representative elsewhere.
- **Off-policy validity** depends on the uniform logging with known propensities.

## Maintenance & periodic review

- **Owner:** Grupo 65 (ML lead). **Cadence:** review this card every retraining
  cycle and at minimum quarterly, or on any drift/guardrail alert.
- Triggers for an out-of-cycle review: a drift alert, a golden-set regression, a
  new offer/channel, or any change to eligibility/suitability rules.
- Linked governance: [`system-card.md`](system-card.md), [`lgpd-plan.md`](lgpd-plan.md),
  [`mlops-plan.md`](mlops-plan.md).
