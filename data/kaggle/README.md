# Kaggle base — source, version & license

> The raw CSV is **not** versioned (see `.gitignore`); download it with the
> instructions below. Provenance is also pinned in code at
> `src/aep/data/source.py` (with a SHA-256 checksum the loader verifies).

- **Dataset:** Bank Marketing (`bank-additional-full.csv`)
- **Kaggle ref:** [`henriqueyamahata/bank-marketing`](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
- **Canonical upstream:** UCI Machine Learning Repository — Bank Marketing
  (<https://archive.ics.uci.edu/dataset/222/bank+marketing>)
- **Version / snapshot:** `bank-additional-full.csv` — 41,188 rows, 20 inputs + target `y`
- **SHA-256 (pinned):** `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) — <https://creativecommons.org/licenses/by/4.0/>
- **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset].
  UCI Machine Learning Repository. <https://doi.org/10.24432/C5K306>

## Known limitations

- Observational telemarketing data from **one Portuguese bank, 2008–2010**; not
  representative of other institutions, products or periods.
- Class-imbalanced (~11% positive).
- Contains **no** client identifiers, income, wealth or protected attributes.

## Download

**Option A — UCI (no credentials):**

```powershell
# Downloads, unzips the nested archives, and copies the CSV into data/kaggle/
$z = "$env:TEMP\bank.zip"
Invoke-WebRequest "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip" -OutFile $z
Expand-Archive $z "$env:TEMP\bankmkt" -Force
Get-ChildItem "$env:TEMP\bankmkt" -Filter *.zip | ForEach-Object { Expand-Archive $_.FullName "$env:TEMP\bankmkt" -Force }
Copy-Item (Get-ChildItem "$env:TEMP\bankmkt" -Recurse -Filter bank-additional-full.csv).FullName .
```

**Option B — Kaggle CLI** (requires a Kaggle account + `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d henriqueyamahata/bank-marketing -p data/kaggle --unzip
# ensure the file is named bank-additional-full.csv in data/kaggle/
```

Then build the processed, leakage-free base and reports:

```bash
uv run aep data all     # build + dictionary + quality   (or ./make.ps1 data)
```

## Temporal-leakage decision

The **`duration`** column (call length in seconds, known only *after* contact)
is dropped before any modelling — it almost perfectly encodes the outcome. The
full rationale and the kept-but-flagged columns are in
[`data_dictionary.md`](data_dictionary.md) and
[`../../reports/data-quality.md`](../../reports/data-quality.md).
