# Which universe supplies the unavailable columns?

The original study draws a uniform mask of size $s$ from all $p$ supplied columns. In this diagnostic a second law draws from the $m$ columns used by at least one split anywhere in the fitted full forest. Each law compares a candidate to the same full forest facing the same mask and normalizes by its own available fraction. These are two distinct physical experiments. Selecting exactly a fraction of all installed channels allows the number of available model inputs to vary. Selecting exactly a fraction of the model's input channels fixes their count.

## Relation between the risks

For an $s$-subset of the declared universe, let $J$ be the number of structurally used inputs it contains. Then $J$ is hypergeometric with population size $p$, $m$ successes and sample size $s$. Conditional on $J=j$, the observed active features form a uniform $j$-subset. Unused inputs have no effect on the response of either forest. Write $V_j^{(m)}$ for expected excess loss divided by $j/m$ and define the zero-size term to be zero. Conditioning on $J$ gives

$$
V_s^{(p)}=\sum_{j\ge 1}\Pr(J=j)\,\frac{p j}{s m}\,V_j^{(m)} .
$$

The weights sum to one since $\mathbb E[J]=sm/p$. Thus the worst excess under the declared law cannot exceed the worst excess under the active law. The endpoints coincide under the two scalings. At declared size one only the active singleton term can contribute, and at full availability all active columns are present. This applies equally to signed losses.

For a threshold reversal at the study's actual budget, take three structurally used features. On a single row use the manuscript's nested path increments $+.03,-.03$, residual $z=1$, and deploy no columns. The only improving active mask is exactly $\{1,2\}$, reducing squared loss by $1-.97^2=.0591$. The maximum excess per available fraction is .02955. Adding one unused column gives .01970 and reverses the .02 decision. The executable check enumerates these masks directly and verifies the mixture identity with rational signed mask losses for all positive sizes through eight declared features.

## Recorded supports and retraced paths

We fixed the protocol before the new calculations. All earlier outcomes had already been released. Original forests, splits, selection guards, nominal index tie order and loss allowances were preserved. Held labels were masked throughout the first stage. All 55 per-fit records, including the newly selected wide supports, were sealed together before the second stage read held labels. The effective universe is derived only from the full fitted trees, including branches unvisited on calibration or test rows.

| Panel | Original arm | Acceptance breaches, declared / active | Held breaches, declared / active | Largest held increase in normalized excess |
|---|---|---:|---:|---:|
| Seven sources, 35 fits | MDI | 0 / 0 | 9 / 9 | 0 |
| Seven sources, 35 fits | MDI + MinPI | 0 / 0 | 10 / 10 | 0 |
| Four wide sources, 20 fits | MDI | 0 / 0 | 3 / 3 | .00000961 |
| Four wide sources, 20 fits | MDI + MinPI | 0 / 0 | 4 / 4 | .00000961 |

No fixed-support decision changed at either stage. The largest observed increase in the acceptance excess was .00026527 on Seoul, seed 43. On the wide panel, with both paths retraced and calibrated in the active universe, every MDI support and 19 of 20 portfolio supports remained unchanged. Oral toxicity, seed 46, retained 177 reachable columns instead of 181. The held quota there had the same disposition. The equal-source geometric portfolio/MDI reachable-width ratio was .891779, compared with .892776 under the original selectors. Under common active-universe scoring the retraced MDI and portfolio had three and four held violations respectively. Intact-data differences for the active portfolio versus MDI averaged −.00317 AUC, +.00007 accuracy, +.00079 R² and +.00014 scaled MAE. Each predictive metric involves two source means.

The report checker reproduced the original narrow-panel outer worst losses for both arms and the wide-panel intact metrics. It tested endpoint equality and the worst-loss inequality on all paired supports, including both acceptance and outer data.

From the thesis checkout, run the following commands in order. Recollection requires the restored source data and earlier structural-support receipts.

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_universe collect
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_universe score
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_universe_report
```

The [protocol](../../../../experiments/minpi_rework/predictive_universe_protocol.md), [hashes](lock.json), [selected supports](supports.json), [paired scores](scores.csv), [fixed-support summaries](fixed-summary.csv), and [20 wide reselections](reselection.csv) record the experiment.
