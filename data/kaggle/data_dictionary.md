# Data dictionary — Bank Marketing (bank-additional-full)

> Generated from `aep.data.schema` (`aep data dictionary`). Do not edit by hand.

Source: `henriqueyamahata/bank-marketing` (Kaggle) / UCI Bank Marketing. Rows: 41,188. Separator: `;`.

| # | Column | Kind | Leakage | Description | Allowed values |
|---|--------|------|---------|-------------|----------------|
| 1 | `age` | numeric | — | Client age in years. |  |
| 2 | `job` | categorical | — | Type of job. | admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown |
| 3 | `marital` | categorical | — | Marital status. | divorced, married, single, unknown |
| 4 | `education` | categorical | — | Education level. | basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown |
| 5 | `default` | categorical | — | Has credit in default? | no, yes, unknown |
| 6 | `housing` | categorical | — | Has a housing loan? | no, yes, unknown |
| 7 | `loan` | categorical | — | Has a personal loan? | no, yes, unknown |
| 8 | `contact` | categorical | — | Contact communication type. | cellular, telephone |
| 9 | `month` | categorical | — | Last contact month of year. | jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec |
| 10 | `day_of_week` | categorical | — | Last contact day of the week. | mon, tue, wed, thu, fri |
| 11 | `duration` | numeric | ⛔ yes | Last contact duration, in seconds. |  |
| 12 | `campaign` | numeric | — | Number of contacts performed during this campaign for this client (includes the last contact). |  |
| 13 | `pdays` | numeric | — | Days since the client was last contacted in a previous campaign (999 = never previously contacted). |  |
| 14 | `previous` | numeric | — | Number of contacts performed before this campaign for this client. |  |
| 15 | `poutcome` | categorical | — | Outcome of the previous campaign. | failure, nonexistent, success |
| 16 | `emp.var.rate` | numeric | — | Employment variation rate (quarterly). |  |
| 17 | `cons.price.idx` | numeric | — | Consumer price index (monthly). |  |
| 18 | `cons.conf.idx` | numeric | — | Consumer confidence index (monthly). |  |
| 19 | `euribor3m` | numeric | — | Euribor 3-month rate (daily). |  |
| 20 | `nr.employed` | numeric | — | Number of employees (quarterly). |  |
| 21 | `y` | target | — | Did the client subscribe a term deposit? | no, yes |

## Dropped columns (temporal / post-contact leakage)

- **`duration`** — Known only AFTER the call ends, so it is unavailable at decision time. It almost perfectly encodes the outcome (duration=0 => y=no). UCI explicitly advises discarding it for realistic models.
