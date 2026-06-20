# Synthetic data generation report

_Generated on 2026-06-20 by `aep synth report`._

## 1. Purpose

The Bank Marketing base has no offers, no channels and no bandit feedback. This layer **synthesizes** a commercial experimentation environment on top of the factual context so we can study exploration vs. exploitation without any real client data. It is **physically separate** from the Kaggle base.

## 2. Process & seeds (reproducible)

- **Master seed:** `42` (sub-seeds: reward_model=`143`, events=`244`, delay=`345`).
- **Impressions:** 20,000 over a **90-day** horizon.
- **Arms:** 8 offers (product x channel x tone) with eligibility rules.
- **Context:** 11 features (age, previous, emp.var.rate, euribor3m, cons.conf.idx, was_prev_contacted, poutcome_success, contact_cellular, edu_university, white_collar, age_senior).
- **Reward funnel:** `click ~ Bernoulli(p_click(context, offer))`, then `conversion ~ Bernoulli(p_conv(context, offer))` only if clicked. `p_conv` is boosted by the client's real subscription signal for deposit-like products (keeping the layer grounded).
- **Logging policy:** uniform over eligible offers, so the recorded `propensity = 1 / n_eligible` is exact (enables unbiased off-policy eval).
- **Delayed rewards:** converters realize after `1 + Poisson(lambda_product)` days; non-converters are censored at **14 days**.

## 3. Summary statistics

- Overall click rate: **35.67%** | conversion rate: **5.93%**.
- Mean eligible arms per impression: **7.16** (of 8).
- Converter delay (days): mean **5.9**, median **6**, max **18**.

### Per-offer exposure and rates

| Offer | Exposure | Click rate | Conversion rate |
|-------|----------|------------|-----------------|
| `O1` | 2,834 | 39.31% | 8.01% |
| `O2` | 2,835 | 26.56% | 2.93% |
| `O3` | 2,836 | 41.96% | 7.97% |
| `O4` | 2,819 | 37.11% | 4.43% |
| `O5` | 2,242 | 44.96% | 6.24% |
| `O6` | 2,873 | 28.12% | 4.32% |
| `O7` | 2,713 | 30.34% | 4.75% |
| `O8` | 848 | 46.23% | 15.57% |

### Per-segment exposure and conversion

| Segment | Exposure | Conversion rate |
|---------|----------|-----------------|
| `mid_nwc` | 7,368 | 4.37% |
| `mid_wc` | 4,837 | 6.10% |
| `young_nwc` | 4,246 | 6.17% |
| `young_wc` | 2,940 | 8.33% |
| `senior_nwc` | 516 | 11.05% |
| `senior_wc` | 93 | 5.38% |

### Oracle optimal arm by segment (why context matters)

The best arm (highest true reward prob, among eligible) **differs across segments** — a context-free policy cannot capture this, motivating the contextual bandit in Stage 3.

| Segment | Dominant optimal offer |
|---------|------------------------|
| `mid_nwc` | `O1` |
| `mid_wc` | `O8` |
| `senior_nwc` | `O7` |
| `senior_wc` | `O7` |
| `young_nwc` | `O1` |
| `young_wc` | `O8` |

## 4. Hypotheses encoded

- Conversion is **heterogeneous by segment** (age, education, occupation, previous-campaign outcome).
- Channels differ in engagement (app_push > cellular > telephone > email for click), with age interactions.
- Premium/investment products suit older, white-collar, educated segments; loans suit younger segments.
- Rewards are **delayed** and **censored**, so naive immediate-reward estimates are biased early (cold-start).

## 5. Limitations & risks

- The oracle is a **modeling assumption**, not reality; absolute rates are not meaningful, only *relative* policy comparisons are.
- Using the real subscription label as receptiveness couples the synthetic layer to one base; do not interpret as causal product demand.
- Uniform logging maximizes off-policy validity but is not how a real system would log; a deployed logger needs its own propensity tracking.
- **Reward hacking risk:** a policy could over-exploit the highest-base arm; Stage 4 guardrails and fairness checks address this.
- No protected attributes are used; `segment` is a synthetic construct for exposure analysis only.

