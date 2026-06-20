# MLOps lifecycle plan (Stage 7)

> How an offer/channel/message hypothesis moves from **experiment** to
> **controlled production** with human approval, monitoring and rollback. The
> mechanics are implemented in `aep.mlops` and demonstrated in
> [`reports/mlops-monitoring.md`](../reports/mlops-monitoring.md).

## 1. Policy versioning

- Every candidate policy is a named **version** (e.g. `linucb-v2`) with its
  hyper-parameters and offline metrics recorded in a registry
  (`aep.mlops.registry.PolicyRegistry`).
- Stages: **`staging` → `production` → `archived`**. Exactly one version is in
  `production` at a time. Promotion archives the incumbent so rollback is trivial.
- Locally the registry is JSON-backed; in the Azure target it is the **Azure ML
  Model Registry** with the same stages, and runs are tracked in **MLflow**
  (Stage 3 already logs every policy as a run).

## 2. Retraining triggers

A retraining run is started when any of these fire:

1. **Scheduled cadence** (e.g. weekly) on fresh logged data.
2. **Data drift alert** — PSI ≥ 0.25 on any context feature.
3. **Reward drift alert** — conversion rate moves ≥ 30% relative between windows.
4. **New hypothesis** — a new arm (offer/channel/message) is added to the catalog.

Drift is monitored continuously by `aep.mlops.drift.monitor_drift` (PSI per
feature + reward-rate change); see the monitoring report for a fired example.

## 3. Promotion gate (automated checks)

Before a human is asked to approve, the candidate must clear
`aep.mlops.promotion.evaluate_promotion`:

| Check | Rule | Type |
|-------|------|------|
| Adversarial guardrails | 100% of adversarial golden cases pass | **hard (never waived)** |
| Golden-set quality | pass rate ≥ 85% | soft |
| Fairness floor | exposure value disparity ≥ 0.20 | soft |
| No conversion regression | ≥ production − 0.5 pp | soft (vs production) |
| No fairness regression | disparity ≥ production − 0.05 | soft (vs production) |

A **hard** failure (e.g. an adversarial suitability breach) blocks promotion
outright. The first policy is bootstrapped against absolute floors only.

## 4. Human approval gate

- The registry refuses `promote()` until an **approver** has signed off
  (`approve(version, approver, notes)`); attempting to promote an unapproved
  version raises `ApprovalError`.
- Sensitive/financial decisioning keeps a **human in the loop**: the reviewer
  inspects the offline report (metrics matrix, golden set, fairness, drift) and
  records who approved and why. This satisfies the LGPD "human in the loop"
  requirement for sensitive decisions (see [`lgpd-plan.md`](lgpd-plan.md)).

## 5. Promotion & rollback

- `promote(version)` moves an approved candidate to `production` and archives the
  previous production version.
- `rollback()` re-promotes the most recently archived version in one step — the
  immediate safety valve if a freshly promoted policy misbehaves in production.
- Because serving loads the **production** version by reference, promotion and
  rollback do **not** require a code redeploy.

## 6. End-to-end procedure (test → approve → promote)

1. **Experiment:** retrain candidate on fresh data; log the run to MLflow.
2. **Evaluate:** run the offline suite (golden set, IPS/SNIPS, fairness) — this
   produces the candidate's metrics (`build_candidate_metrics`).
3. **Register:** add the candidate to the registry in `staging`.
4. **Gate:** run the automated promotion checks; a hard failure stops here.
5. **Approve:** a human reviewer signs off (approval gate).
6. **Promote:** move to `production`; the incumbent is archived.
7. **Monitor:** watch drift, reward and exposure dashboards.
8. **Rollback** if needed.

The demo in `reports/mlops-monitoring.md` walks `linucb-v2` through steps 1–8,
including a rollback to `linucb-v1`.

## 7. Azure mapping

| Local | Azure target |
|-------|--------------|
| JSON registry + stages | Azure ML Model Registry (staging/production) |
| MLflow file store | Azure ML managed MLflow |
| Promotion gate (code) | Azure ML pipeline gate + branch policies |
| Approval gate | Azure DevOps/GitHub environment approvals |
| Drift monitor | Azure ML data drift monitors + Azure Monitor alerts |
| Retrain trigger | Azure ML pipeline scheduled / event-triggered |

## 8. Commands

```bash
aep mlops report            # run the lifecycle demo + drift, write the report
aep mlops status            # show registry versions/stages/production
aep mlops drift --shock both  # exercise the drift detectors
aep mlops approve linucb-v2 --approver you   # approval gate
aep mlops promote linucb-v2 # promote (blocked unless approved)
aep mlops rollback          # revert to the previous production version
```
