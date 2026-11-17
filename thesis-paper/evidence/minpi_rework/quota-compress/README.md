# Directional quotas for teacher-fidelity compression

The seed-42 study fit one linear additive teacher per historical source. A retained subset had to approximate the full centered score vector on separate calibration rows within 20% of its Euclidean norm. It used the same trained functions on the selected inputs at deployment. The six original sources and eight previously opened UCI sources were fixed in advance. The [protocol](../../../../experiments/minpi_rework/quota_compress_protocol.md), [sealed selection receipts](selection-receipts.zip) and [per-source outcomes](historical.csv) identify each support and the calibrated and outer results. Predictions are under the ignored worktree results directory.

A unit vector in the space of calibration scores votes for a candidate whenever the directional residual is at most the fidelity budget. The candidate's own normalized residual rejects it whenever its norm exceeds the budget. At most K columns can contribute their K largest positive ballot weights. If even that maximum fails after conditioning on inclusion, the column must be absent from every qualifying support of this width or smaller. Conditioning on exclusion gives the corresponding mandatory-inclusion test. These signs describe necessity for a specified score vector and budget.

## Cardinality and certification

| Source | Inputs | Gain order | Energy order | Backward greedy | Ballot master with fixings | Master lower bound | Native cone | Bidirectional local search |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Madelon | 100 | 85 | 55 | 53 | 53 | 33 | 53 | 52 |
| Spambase | 56 | 28 | 26 | 25 | 25 | 22 | 23 | 23 |
| Abalone | 8 | 7 | 4 | 6 | 4 | 4 | 4 | 4 |
| Gisette | 100 | 99 | 39 | 32 | 32 | 15 | 24 | 31 |
| Credit card fraud | 29 | 26 | 12 | 12 | 12 | 12 | 12 | 12 |
| ISOLET | 100 | 97 | 94 | 92 | 92 | 56 | 92 | 92 |
| Glass | 9 | 9 | 8 | 8 | 8 | 8 | 8 | 8 |
| Raisin | 7 | 7 | 5 | 5 | 5 | 5 | 5 | 5 |
| WiFi | 7 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| Leaves | 64 | 60 | 59 | 59 | 59 | 46 | 59 | 59 |
| Airfoil | 5 | 5 | 4 | 4 | 4 | 4 | 4 | 4 |
| Yacht | 6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Slump | 7 | 7 | 5 | 5 | 5 | 5 | 5 | 5 |
| Energy | 8 | 8 | 4 | 4 | 4 | 4 | 4 | 4 |

The ballot master with and without explicit fixings returned identical widths on all 14 sources. Both certified the optimal width on nine. The native conic solver certified eleven. The three methods consumed 51.85 s, 52.09 s and 38.69 s respectively in their optimization components, with a nominal ten-second limit per source and final solver shutdown overhead. SCIP certificates use its numerical tolerances. Hundreds of small signed-capacity and exact-fit cases were independently enumerated. The secondary bidirectional local search took 1–16 ms per source after forming the Gram matrix. It equalled or improved the ballot incumbent on every source. [Local-control results](local-control.csv)

Before the integer solver, necessary signed conditions identified 26 mandatory inclusions across seven sources and 11 exclusions across four. Four of the five sources with more than 50 columns yielded no initial certified signs. A solver-free follow-up tested smaller capacities directly and included directions from failed single deletions. It certified the local support's minimum width on Abalone, Raisin, WiFi, Airfoil, Yacht and Energy. On the other eight sources it provided lower bounds. The added boundary directions tightened four lower bounds without adding a completed certificate. [Capacity checks](early-certificates.csv)

## Outer prediction

Compared with its own complete linear teacher, the ballot-selected model averaged −0.00321 AUC and −0.00813 accuracy on nine classification sources. The five regression sources averaged −0.03473 R² and +0.01847 development-standardized MAE. The mean R² loss exceeded the project's 0.02 allowance. Accuracy fell by 0.0195 on Spambase and 0.0250 on Leaves. Airfoil lost 0.0859 R². Exact calibration fidelity does not guarantee these outer metrics.

Gain in the table ranks columns of the *same linear deployment* using a separately trained XGBoost model. The archived Gain path with XGBoost refit on retained columns is a stronger nonlinear prediction pipeline with a different deployment and training pool. On the original Madelon split it attained 0.8696 AUC with 13 columns, versus 0.5944 with 53 for the ballot-trimmed linear teacher. On ISOLET the corresponding accuracies were 0.8135 with 50 columns and 0.6962 with 92. [Archived nonlinear receiver](../../minpi_compression/legacy-gain/xgb-receiver/scores.csv)

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.quota_compress_check
.\thesis-python.ps1 -m experiments.minpi_rework.quota_compress_run select
.\thesis-python.ps1 -m experiments.minpi_rework.quota_compress_run seal
.\thesis-python.ps1 -m experiments.minpi_rework.quota_compress_run score
.\thesis-python.ps1 -m experiments.minpi_rework.quota_compress_control run
.\thesis-python.ps1 -m experiments.minpi_rework.quota_compress_control score
.\thesis-python.ps1 -m experiments.minpi_rework.quota_compress_early check
.\thesis-python.ps1 -m experiments.minpi_rework.quota_compress_early run
```
