# MLOps monitoring report (Stage 7)

_Generated on 2026-06-20 by `aep mlops report`._

## 1. Lifecycle demo — experiment to controlled production

A new offer-policy hypothesis (`linucb-v2`, more exploration) goes from experiment to production through the automated gate **and** a human approval, then is rolled back — exactly the controlled path required.

| Step | Detail |
|------|--------|
| register+promote v1 | bootstrap gate passed=True; v1 -> production |
| register v2 (staging) | candidate metrics: {'conversion': 0.13812, 'snips': 0.13699, 'golden_pass_rate': 0.9091, 'adversarial_pass': 1.0, 'fairness_disparity': 0.2878} |
| promotion gate v2 | passed=True; adversarial_guardrails=OK; golden_pass_rate=OK; fairness_floor=OK; no_conversion_regression=OK; no_fairness_regression=OK |
| approve+promote v2 | v2 -> production, v1 -> archived |
| rollback | production reverted to linucb-v1 |

Final production policy after the demo (post-rollback): **`linucb-v1`**.

Registry stages and approvals:

| Version | Stage | Approved by | Conversion | Golden | Fairness |
|---------|-------|-------------|-----------:|-------:|---------:|
| `linucb-v1` | production | bootstrap | 13.81% | 91% | 0.29 |
| `linucb-v2` | archived | ml-reviewer | 13.81% | 91% | 0.29 |

## 2. Drift monitoring (PSI per context feature)

Thresholds: PSI >= 0.1 warn, PSI >= 0.25 alert. The natural stream is stationary, so the baseline shows no drift; the shocked window (age/euribor shift + halved reward) confirms the detectors fire.

| Feature | PSI (baseline) | PSI (shocked) |
|---------|---------------:|--------------:|
| `age` | 0.001 | 4.060 |
| `previous` | 0.000 | 0.000 |
| `emp.var.rate` | 0.001 | 0.001 |
| `euribor3m` | 0.002 | 10.104 |
| `cons.conf.idx` | 0.002 | 0.002 |
| `was_prev_contacted` | 0.000 | 0.000 |
| `poutcome_success` | 0.000 | 0.000 |
| `contact_cellular` | 0.000 | 0.000 |
| `edu_university` | 0.000 | 0.000 |
| `white_collar` | 0.000 | 0.000 |
| `age_senior` | 0.000 | 0.000 |

- Baseline: max PSI **0.002** (data alert: no).
- Shocked: max PSI **10.104** (data alert: YES).

## 3. Reward drift

| Window | Reference conv. | Current conv. | Rel. change | Alert |
|--------|----------------:|--------------:|------------:|:-----:|
| baseline | 5.87% | 5.99% | +2.0% | no |
| shocked | 5.87% | 3.08% | -47.6% | YES |

## 4. How this maps to the lifecycle

- A drift alert (data or reward) **triggers a retraining run**; the candidate is registered in staging and must clear the promotion gate.
- The promotion gate enforces hard guardrails (adversarial suitability), golden-set quality and no conversion/fairness regression.
- A **human approver** signs off before production; **rollback** is one call back to the previous archived version.
- Experiments and policy versions are tracked in MLflow (Stage 3 logs runs; the Azure target uses the Azure ML Model Registry stages).

See [`docs/mlops-plan.md`](../docs/mlops-plan.md) for the full policy.

