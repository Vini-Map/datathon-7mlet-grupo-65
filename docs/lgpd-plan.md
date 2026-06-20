# LGPD plan — data protection & privacy

> Academic/demonstration plan for the 7MLET Datathon (Grupo 65). The project uses
> **no real personal data**; this documents how the design *would* comply with
> the LGPD (Lei Geral de Proteção de Dados, Lei nº 13.709/2018) if operated.

## 1. Legal basis (base legal)

- The factual base (Bank Marketing) is a **public, anonymized research dataset**
  (CC BY 4.0); the experimentation layer is **synthetic**. No personal data of
  real data subjects is processed in the demo.
- In a real deployment, the applicable bases (Art. 7) would be **legitimate
  interest** (Art. 7, IX) for non-intrusive offer optimization and/or **consent**
  (Art. 7, I) for direct marketing, with an opt-out always available. A
  **Legitimate Interest Assessment (LIA)** and, given automated decisioning, a
  **DPIA/RIPD** would be required before go-live.

## 2. Purpose (finalidade)

- Single, explicit purpose: **selecting which eligible offer/channel/message to
  present** to a client, to improve relevance. No profiling for unrelated
  purposes; no sale of data; no sensitive-attribute inference.

## 3. Data minimization (minimização)

- The model uses **only non-sensitive context features** (age, prior-contact
  signals, macro indicators, occupation/education-derived segment flags).
- **Explicitly excluded:** identifiers, income, wealth, gender, race, religion,
  health, political/union membership and any Art. 5 sensitive data.
- The leakage column `duration` is dropped (also avoids post-hoc data not
  available at decision time).

## 4. Mapping of identifiers & protected attributes

| Category | In the system? | Handling |
|----------|----------------|----------|
| Direct identifiers (name, CPF, account) | **No** | Never collected/stored. |
| Sensitive data (Art. 5, II) | **No** | Excluded by design; not used as features. |
| Quasi-identifiers (age, job, education) | Yes (coarse) | Used only as model context; bucketed into synthetic segments; not exposed in assistant prompts beyond a brief non-sensitive summary. |
| Behavioral (offer, click, conversion) | Synthetic only | Used for learning/audit; retained per §6. |

## 5. Logs & telemetry policy

- The **decision audit log** records: audit id, timestamp, selected offer, reason
  codes, policy version, and the **non-sensitive** context. It is the basis for
  the data subject's right to **explanation/review of automated decisions**
  (Art. 20).
- Telemetry (latency, errors, drift metrics) is **aggregate** and carries no
  personal data. In Azure, logs live in Log Analytics/PostgreSQL with access
  controlled by Entra ID RBAC and secrets in Key Vault.

## 6. Retention cycle (retenção)

- **Audit/decision records:** retained for a defined period (e.g. 12–24 months)
  to support auditability and dispute resolution, then deleted or anonymized.
- **Synthetic experimentation data:** retained while the experiment is active;
  archived to cold storage and purged on a schedule.
- Retention periods are configurable and reviewed at each governance review.

## 7. Data-subject rights

- Access, correction, deletion, and **review of automated decisions** (Art. 20)
  are supported by the audit log + human-in-the-loop review path. Opt-out of
  marketing is honored as an eligibility rule.

## 8. Human in the loop

- Sensitive/financial decisions keep a **human reviewer**; policy promotion
  requires human approval (Stage 7 approval gate). Fully automated adverse
  decisions are out of scope for this demo.

## 9. Incident response plan

1. **Detect** — alerts (drift, guardrail breach, anomalous access, data-handling
   error).
2. **Contain** — roll back the policy (one call) and/or disable the affected
   endpoint via API Management.
3. **Assess** — scope and severity; whether any personal data was involved.
4. **Notify** — in a real deployment, notify the **ANPD** and affected subjects
   when required (Art. 48), within the controller's defined SLA.
5. **Remediate & review** — fix root cause, update guardrails/cards, record
   lessons learned.

## 10. Governance & review

- **Owner:** Grupo 65 (acting DPO/ML lead). **Cadence:** review this plan
  quarterly and on any incident or scope change.
- Linked: [`model-card.md`](model-card.md), [`system-card.md`](system-card.md),
  [`mlops-plan.md`](mlops-plan.md).
