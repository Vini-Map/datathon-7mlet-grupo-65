# Kaggle base — source, version & license

> **Stage 1 deliverable.** This file documents the factual reference dataset.
> The raw CSV is **not** versioned (see `.gitignore`); it is downloaded via the
> data loader / script below.

- **Dataset:** Bank Marketing
- **Author / source:** `henriqueyamahata/bank-marketing` (Kaggle)
- **Link:** _to be filled in Stage 1_
- **Version / snapshot date:** _to be filled in Stage 1_
- **License:** _to be filled in Stage 1_
- **Known limitations:** _to be filled in Stage 1_

## Download

_Download instructions (Kaggle CLI / script + placeholder) will be added in Stage 1._

## Temporal-leakage decision

The `duration` column (call duration, only known **after** contact) and any other
post-contact fields will be dropped, with justification, during Stage 1.
