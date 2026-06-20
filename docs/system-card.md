# System Card — AEP decision service + assistant

> Academic/demonstration system for the 7MLET Datathon (Grupo 65). Describes the
> whole decisioning system (not just the model): scope, flow, dependencies,
> guardrails, risk scenarios and monitoring. **Not** production-ready.

## 1. Scope

A service that, given an **eligible client context**, selects an offer via a
multi-armed-bandit policy and returns it with reason codes, the policy version
and an auditable record. An LLM+RAG assistant explains decisions, retrieves
synthetic policy and summarizes experiments. All experimentation data is
synthetic; the factual base carries no personal identifiers.

## 2. Decision flow

1. Request arrives (`POST /decide`) with a non-sensitive context.
2. Eligibility/suitability mask is computed; ineligible offers are removed.
3. The **production** policy (frozen LinUCB) scores eligible arms and selects one.
4. The decision is written to an **auditable log** (reason codes, arm, policy
   version, timestamp, audit id).
5. Optionally, the assistant explains the decision grounded in retrieved policy.

## 3. Dependencies

- **Decision:** processed Bank Marketing base, synthetic offer catalog +
  eligibility rules, frozen LinUCB policy (from the policy registry).
- **Assistant:** `LLMProvider` (offline MockProvider by default; Anthropic
  optional → Azure OpenAI in target) + a TF-IDF RAG retriever over synthetic
  policy docs (→ Azure AI Search in target).
- **Platform (target):** Azure Container Apps, API Management, Azure ML registry,
  PostgreSQL/ADLS, Key Vault, Entra ID, Azure Monitor (see `architecture-azure.md`).

## 4. Guardrails

- **Eligibility/suitability is enforced first:** an ineligible offer is never
  selected regardless of estimated reward (e.g. no credit/loan/investment for a
  defaulting client; Premium Advisory restricted to white-collar, age ≥ 30). All
  12 adversarial golden cases pass.
- **Deterministic serving:** the production policy scores by mean only; decisions
  are reproducible and auditable.
- **Approval gate + hard checks** before any policy reaches production (Stage 7).
- **Assistant grounding:** the assistant answers only from retrieved policy and
  decision facts, is instructed never to invent client data, and runs offline by
  default (no data leaves the host).
- **No sensitive features** enter the model or the prompts.

## 5. Risk scenarios & mitigations

| Risk | Scenario | Mitigation |
|------|----------|------------|
| **Reward hacking** | A policy over-exploits one high-base arm, ignoring suitability/diversity. | Exposure-concentration monitoring; fairness floor in the promotion gate; regret/optimality tracking. |
| **Context manipulation** | Crafted input tries to elicit a restricted offer (e.g. premium for an ineligible client). | Eligibility mask applied before scoring; adversarial golden cases assert the offer is never shown. |
| **Assistant abuse** | Prompt tries to extract data or produce non-compliant advice. | Grounded RAG-only answers, strict system prompt, no sensitive data in context, offline default provider. |
| **Suitability violation** | A defaulting/underage client is offered an unsuitable product. | Hard eligibility rules + hard adversarial check in the promotion gate (never waived). |
| **Silent degradation** | Data/reward drift erodes performance unnoticed. | PSI + reward-drift monitors with alerts that trigger retraining (Stage 7). |
| **Bad promotion** | A worse policy reaches production. | Automated gate + human approval; one-call rollback. |

## 6. Monitoring plan

- **Online:** decision audit log; request/latency/error telemetry (Azure Monitor
  in target).
- **Quality:** periodic golden-set runs; off-policy value tracking.
- **Drift:** PSI per context feature and reward-rate change between windows;
  alerts at PSI ≥ 0.25 or ≥ 30% relative reward change.
- **Fairness:** exposure share and delivered value per segment; alert on
  concentration or value-disparity regressions.

## 7. Human oversight & review

- **Human in the loop** for promotion (approval gate) and for any sensitive
  decision review.
- **Owner:** Grupo 65. **Review cadence:** this card is reviewed every retraining
  cycle and at minimum quarterly, and on any guardrail/drift alert.
- Linked: [`model-card.md`](model-card.md), [`lgpd-plan.md`](lgpd-plan.md),
  [`mlops-plan.md`](mlops-plan.md).
