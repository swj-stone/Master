# Joint empirical-background sensitivity

The two external panels use a fixed, fitted XGBoost voter and up to eight half-width candidates per fold. This retrospective experiment changes how the voter supplies omitted features. For each proposal row, the visible coordinates stay fixed. Each of 16 complete fitting rows in turn supplies all omitted coordinates, preserving their observed joint values within that donor. The classifier's 16 probability vectors are averaged before any metric is evaluated. This removes dependence between the retained row and its donor coordinates. It does not estimate the conditional distribution of missing features given observed ones. Joint marginal integration is a missing-feature convention discussed by [Covert, Lundberg and Lee (2020)](https://proceedings.neurips.cc/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html). This experiment measures subset performance directly and computes no Shapley values.

The existing candidate/layout locks determine the 16 tasks and 46 folds. Separate random streams produce donor samples A and B without using labels to choose donors. Each sample is shared across all candidates and restored contexts in that fold. The score for point uses one panel and no restoration. The full score uses the third-largest of four panel margins and takes its minimum across the original separating contexts. Both retain the same Gain-based tie order. The design was recorded locally at `6cd282b` and all donor scores were recorded at `7f03919` before joining the nominees to archived outcomes. Those outcomes had already been studied.

## Safe compressed fraction

Entries are percent of anchor columns saved while satisfying the measured outer tolerances, averaging within task, then across tasks. Each selection receives the original one-candidate guard. Rejected selections return the anchor. The logistic receiver uses its own guard, refit and anchor scores on the same supports.

| Receiver and panel | Median point / full | Donor A point / full | Donor B point / full | Exact uniform |
|---|---:|---:|---:|---:|
| XGBoost, ten tasks | 7.29 / 7.29 | 4.85 / 6.18 | 6.18 / 6.18 | 5.49 |
| XGBoost, six wide tasks | 7.34 / 7.34 | 4.63 / 10.07 | 10.07 / 4.57 | 7.07 |
| Logistic, ten tasks | 5.78 / 5.78 | 3.33 / 4.67 | 4.67 / 4.67 | 6.05 |
| Logistic, six wide tasks | 8.21 / 10.83 | 5.50 / 10.78 | 19.16 / 19.16 | 10.99 |

On the six wide tasks the XGBoost full-minus-point change was +5.44 points for sample A and −5.50 for B. The respective descriptive 95% task-bootstrap intervals were [0, 11.00] and [−11.06, 0]. The two full-rule nominees agreed on 19 of 20 actionable folds on the first panel and 9 of 15 on the wide panel. The two point-rule nominees agreed on 13 of 20 and 10 of 15. Two draws expose sensitivity to a Monte Carlo approximation at this budget. They cannot establish the behavior of exact marginal integration or estimate the full variation across draws.

Each wide-panel draw required 201,728 model rows for point and 2,126,768 for full. Nomination took 0.87 s for point and 10.14–10.32 s for full in these runs. Counts include scoring the reference supports. These measurements exclude the common initial model and the guard and final fits. Singleton folds performed no nomination. The first-panel counts were 262,992 and 1,603,040 rows per draw.

## Reproduction

From the thesis checkout, `.\thesis-python.ps1 -m experiments.context_review.marginal_background` uses the original split feature matrices to recompute the nominations. It first checks all archived median scores, validates batched class-probability averaging against direct donor-by-donor predictions, and seals all supports and scores in [the nomination lock](../../thesis-paper/evidence/marginal/nomination-lock.json).

After the nominations are present, `.\thesis-python.ps1 -m experiments.context_review.marginal_replay` uses the sealed guard and outer results to reconstruct the comparison, including every per-fold policy outcome. It also checks the two median controls and exact uniform against their earlier independent records to absolute tolerance 1e−12. [Task results](../../thesis-paper/evidence/marginal/tasks.csv) · [Paired contrasts](../../thesis-paper/evidence/marginal/contrasts.csv) · [Nominee agreement](../../thesis-paper/evidence/marginal/agreement.csv) · [Costs](../../thesis-paper/evidence/marginal/costs.csv)
