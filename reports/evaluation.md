# Offline evaluation report (Stage 4)

_Generated on 2026-06-20 by `aep eval report`. Policy: `linucb-v1` (frozen LinUCB, deterministic)._

## 1. Metrics matrix (headline)

| Metric | Value | Reading |
|--------|------:|---------|
| Golden-set pass rate | **90.9%** | behavioral correctness on 22 cases |
| Off-policy value (SNIPS) | 13.70% | estimated conversion from logged data |
| Off-policy value (IPS) | 13.39% | unbiased but higher-variance |
| Oracle-true value | 13.81% | true policy conversion (synthetic) |
| Logging policy value | 5.93% | uniform-random baseline |
| Optimal (oracle) value | 14.47% | upper bound |
| Relative lift vs logging | **+133%** | policy vs uniform |
| Exposure value disparity | 0.29 | min/max segment value (1=equal) |

The SNIPS/IPS estimates (from logged data only) land close to the oracle-true value, validating the off-policy methodology.

## 2. Justification of metrics

- **Off-policy value (IPS/SNIPS)** is the right primary metric: the logging policy is uniform-over-eligible with exact propensities, so we can estimate a new policy's value **without re-running** the system. SNIPS reduces IPS variance at the cost of small bias.
- **Regret / % optimal** (Stage 3) measure decision quality online; here we focus on **value** and **behavioral** correctness.
- Raw accuracy is intentionally avoided: rewards are rare (~imbalanced).

## 3. Golden set (versioned)

`data/golden_set/evaluation_cases.jsonl` — **22 cases** with explicit context, expected action, expected reward and pass criteria. Coverage: typical, per-segment, edge and adversarial.

Pass rate by category:

| Category | Pass rate |
|----------|----------:|
| adversarial | 100% |
| edge | 100% |
| segment | 83% |
| typical | 80% |

All **adversarial** suitability guardrails pass (the policy never selects an ineligible or forbidden offer). Failing cases (if any) are documented below and feed the 'when not to use' guidance.

| Case | Category | Expected | Chosen | Why it failed |
|------|----------|----------|--------|---------------|
| `segment_mid_nwc` | segment | O1 | O3 | reward frac 0.51 < 0.7 |
| `typical_with_loans` | typical | O5 | O3 | reward frac 0.58 < 0.7 |

## 4. Sensitivity analysis (serving alpha)

Refitting the serving policy across exploration strengths shows the offline behavior is stable, not an artifact of one hyper-parameter:

| alpha | Golden pass rate | Oracle-true value |
|------:|-----------------:|------------------:|
| 0.1 | 90.9% | 13.81% |
| 0.5 | 90.9% | 13.81% |
| 1.0 | 90.9% | 13.81% |
| 2.0 | 90.9% | 13.81% |

## 5. Exposure fairness across segments

- Largest single-offer exposure share: **23.7%**.
- Expected-conversion disparity (min/max across segments): **0.29**.

Expected conversion delivered per segment:

| Segment | Expected conversion |
|---------|--------------------:|
| `senior_nwc` | 33.27% |
| `senior_wc` | 26.91% |
| `young_wc` | 19.60% |
| `mid_wc` | 15.34% |
| `young_nwc` | 12.76% |
| `mid_nwc` | 9.57% |

Offer exposure share by segment (rows sum to 1):

```
offer           O1      O2      O3      O4      O5      O6      O7      O8
segment                                                                   
mid_nwc     0.3006  0.0186  0.1185  0.0034  0.2492  0.0000  0.3097  0.0000
mid_wc      0.0957  0.0031  0.0076  0.0000  0.3256  0.0010  0.0678  0.4991
senior_nwc  0.0097  0.1105  0.0000  0.0000  0.0000  0.0000  0.8798  0.0000
senior_wc   0.0323  0.0538  0.0000  0.0000  0.0000  0.0000  0.7419  0.1720
young_nwc   0.3907  0.0033  0.4595  0.0212  0.1253  0.0000  0.0000  0.0000
young_wc    0.1364  0.0003  0.2412  0.0082  0.0102  0.0014  0.0000  0.6024
```

## 6. Limitations, biases and when NOT to use

- **Synthetic rewards.** All values come from a modeled oracle; absolute numbers are not real demand. Only relative comparisons are meaningful.
- **Linear-model bias.** The frozen LinUCB can mis-route some segments (see failing golden cases), over-selecting broadly-appealing offers. Do **not** deploy for those segments without monitoring/human review.
- **Exposure concentration.** If one offer dominates exposure, audit for suitability and diversify before any rollout.
- **Off-policy validity** holds only because logging was uniform with known propensities; a real logger must record its own propensities.
- **Not for production-regulated use.** No protected attributes, no real clients; segments are synthetic operational constructs.

