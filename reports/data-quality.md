# Data quality report — Bank Marketing

_Generated on 2026-06-20 by `aep data quality`._

## 1. Provenance

- **Dataset:** Bank Marketing (bank-additional-full)
- **Kaggle ref:** `henriqueyamahata/bank-marketing`
- **Upstream:** https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) (https://creativecommons.org/licenses/by/4.0/)
- **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- **Pinned SHA-256:** `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`
- **On-disk SHA-256:** `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8` (✅ matches)

## 2. Shape & integrity

- Rows: **41,188**
- Columns (raw): **21** → (processed, leakage-free): **20**
- Exact duplicate rows: **12**
- True NaN cells: **0** (categoricals use the literal token `unknown` instead of NaN)

## 3. Target balance

- Positive class (`subscribed = 1`): **4,640** of 41,188 = **11.27%**
- The base is **imbalanced** (~1 in 9). Evaluation must not rely on raw accuracy; use balanced metrics and report per-segment exposure.

## 4. Missingness (`unknown` token by column)

| Column | # unknown | % |
|--------|-----------|---|
| `default` | 8,597 | 20.87% |
| `education` | 1,731 | 4.20% |
| `housing` | 990 | 2.40% |
| `loan` | 990 | 2.40% |
| `job` | 330 | 0.80% |
| `marital` | 80 | 0.19% |

- `pdays = 999` (never previously contacted): **39,673** rows (96.32%). Treated as a sentinel, not a real day count.

## 5. Temporal / post-contact leakage decision

Dropped before any modelling (unavailable at decision time):

- **`duration`** — Known only AFTER the call ends, so it is unavailable at decision time. It almost perfectly encodes the outcome (duration=0 => y=no). UCI explicitly advises discarding it for realistic models.

Kept but flagged: `campaign` includes the current contact; `pdays`, `previous`, `poutcome` describe the *previous* campaign (not leakage). Socio-economic indicators (`emp.var.rate`, `cons.*`, `euribor3m`, `nr.employed`) are macro context known at decision time.

## 6. Numeric summary (raw)

```
             age   duration   campaign      pdays   previous  emp.var.rate  cons.price.idx  cons.conf.idx  euribor3m  nr.employed
count  41188.000  41188.000  41188.000  41188.000  41188.000     41188.000       41188.000      41188.000  41188.000    41188.000
mean      40.024    258.285      2.568    962.475      0.173         0.082          93.576        -40.503      3.621     5167.036
std       10.421    259.279      2.770    186.911      0.495         1.571           0.579          4.628      1.734       72.252
min       17.000      0.000      1.000      0.000      0.000        -3.400          92.201        -50.800      0.634     4963.600
25%       32.000    102.000      1.000    999.000      0.000        -1.800          93.075        -42.700      1.344     5099.100
50%       38.000    180.000      2.000    999.000      0.000         1.100          93.749        -41.800      4.857     5191.000
75%       47.000    319.000      3.000    999.000      0.000         1.400          93.994        -36.400      4.961     5228.100
max       98.000   4918.000     56.000    999.000      7.000         1.400          94.767        -26.900      5.045     5228.100
```

## 7. Limitations

- Observational telemarketing data (2008-2010, one Portuguese bank); not representative of other institutions, products or periods.
- No client identifiers, income, wealth or protected attributes are used.
- Real outcomes seed the **synthetic** reward layer (Stage 2); they are not used as production labels.

