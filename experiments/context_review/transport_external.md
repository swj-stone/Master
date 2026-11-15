# Ten-task UCI transport experiment

The [design](../../configs/context_review/transport_external.json) froze ten tasks, learner, fold seeds, candidate pools, four nomination policies and two primary paired contrasts in commit `f441c21`. The protocol defines the primary outcome as the fraction of anchor columns saved when the refitted model remains within the joint outer tolerance. The policies include the Gain prefix, the archived worst-context vote, its continuous majority margin and the continuous margin on the original panel with no restored features. Commit `84cfdd5` sealed every selection and outer prediction before scoring. The companion study sealed its predictions in `0784efa` before either study opened outer outcomes.

## Reproduction

Run this command from the thesis checkout. It downloads any missing published archives, checks their digests, reconstructs the partitions, independently fits all four policies and verifies the published predictions and metrics.

```powershell
.\thesis-python.ps1 -m experiments.context_review.transport_replay
```

The replay writes into `results/context_review/transport-replay/`. Its comparison covers the 30 probability archives, every nominee and final support, mask and fit counts, and the outer losses. It also independently checks every frozen candidate score against weighted scikit-learn AUC/accuracy calculations and repeats the four-feature Mammographic sensitivity. The original results used Python 3.13.9, NumPy 2.4.6, pandas 3.0.3, SciPy 1.18.0, scikit-learn 1.9.0 and XGBoost 3.3.0. The original timing experiment bound a single logical CPU and rotated arm order across folds. Each end-to-end measure includes fitting, nominating, checking the guard and final prediction on preloaded data. Offline audits and analysis are excluded from timing.

The [prediction seal](../../thesis-paper/evidence/transport/prediction-lock.json) records literal source-file hashes on the original Windows checkout. Git line-ending conversions can change those hashes. Fresh runs generate their own source receipts in the replay directory. Predicted probability archives, selected columns and metrics supply the cross-run comparison. Original selection receipts also contain machine-dependent clock readings. The [fold manifest](../../thesis-paper/evidence/transport/fold-manifest.csv) identifies each split and the corresponding probability archive. Raw archives, fit inputs, test labels, split indices, prediction arrays and per-fold receipts reside in the ignored data and results directories. The source and seeds reconstruct each original row ID.

## Sources and scope

The original conference inputs, 14-task PMLB panel, grouped HAR/Optdigits/drift studies, linear transfer protocol and input directories were checked for previously used datasets and aliases. Both scikit-learn Wine and Wisconsin Diagnostic Breast Cancer had already been evaluated in the thesis's linear transfer experiment. The ten selected sources extend the benchmark with different data. Their convenience sampling and partly shared measurement domains limit inference to this panel. Rice and Dry Bean concern distinct sample collections from related crop-imaging research. The sensitivity excluding both is included in the paired contrasts.

| Source | UCI ID | Rows | Columns | Dataset citation |
|---|---:|---:|---:|---|
| Banknote Authentication | 267 | 1,372 | 4 | [Lohweg, 2012](https://doi.org/10.24432/C55P57) |
| Blood Transfusion Service Center | 176 | 748 | 4 | [Yeh, 2008](https://doi.org/10.24432/C5GS39) |
| Dry Bean | 602 | 13,611 | 16 | [UCI, 2020](https://doi.org/10.24432/C50S4B) |
| HTRU2 | 372 | 17,898 | 8 | [Lyon, 2015](https://doi.org/10.24432/C5DK6R) |
| Mammographic Mass | 161 | 961 | 5 | [Elter, 2007](https://doi.org/10.24432/C53K6Z) |
| QSAR biodegradation | 254 | 1,055 | 41 | [Mansouri et al., 2013](https://doi.org/10.24432/C5H60M) |
| Rice (Cammeo and Osmancik) | 545 | 3,810 | 7 | [UCI, 2019](https://doi.org/10.24432/C5MW4Z) |
| Seeds | 236 | 210 | 7 | [Charytanowicz et al., 2010](https://doi.org/10.24432/C5H30K) |
| Steel Plates Faults | 198 | 1,941 | 27 | [Buscema et al., 2010](https://doi.org/10.24432/C5J88N) |
| Vertebral Column | 212 | 310 | 6 | [Barreto and Neto, 2005](https://doi.org/10.24432/C5K89B) |

We used each published record once. Some held-out records had exactly the same features as a development row in their fold. Aggregated over the three held-out partitions, this occurred for 438/961 Mammographic records, 283/748 Blood, 86/13,611 Dry Bean, 29/1,372 Banknote and 4/1,055 QSAR records. The remaining five tasks had no exact matches. These are record-level prediction results. Matching measured attributes can occur for different people or specimens. Performance in unseen clinics, instruments or laboratories remains unmeasured.

## Results

All ten tasks completed three folds. Ten folds had one-column anchors, including every fold of HTRU2 and Rice. The resulting 20 actionable folds offered 106 distinct fold–candidate pairs. The archived vote assigned zero approvals to every candidate on 16 actionable folds. Both continuous scores produced a unique winner in all 20. Separate weighted scikit-learn calculations reproduced all 106 candidate rankings at the top, with largest absolute discrepancy below 2.7e-14 across the two signed scores.

| Policy | Safe fraction of anchor saved | Guard accepts | Outer breaches | AUC | Accuracy | Extra masks per fold |
|---|---:|---:|---:|---:|---:|---:|
| Gain prefix | 0.00% | 4 | 4 | 0.92065 | 86.46% | 0.00 |
| Archived vote | 1.33% | 5 | 4 | 0.92020 | 86.41% | 5.90 |
| Worst-context continuous margin | 7.29% | 9 | 4 | 0.91829 | 85.98% | 25.33 |
| Empty-context one-panel margin | 7.29% | 7 | 2 | 0.92013 | 86.43% | 4.20 |

Across equally weighted tasks, the archived vote improved safe savings by 1.33 percentage points over Gain, with a 95% task-bootstrap interval of [0.00, 4.00]. The full continuous margin improved safe savings by 7.29 points [1.52, 14.45]. This gain came from five folds across Dry Bean, Mammographic, Seeds and Steel. Omitting Rice and Dry Bean reduced the estimate to 4.95 points. The two margin policies produced identical safe savings on every fold. The full rule changed six nominees relative to the single-panel rule and two final supports. Both additional guard acceptances breached the outer tolerance. The recorded single-run policy wall times and mask counts can be inspected in the [fold results](../../thesis-paper/evidence/transport/fold-results.csv). All policies used one candidate guard refit whenever a deletion was possible.

The locked Mammographic analysis includes the physician's BI-RADS assessment together with age, lesion shape, margin and density. The repository lists BI-RADS in its feature table and reserves that assessment as a physician comparator in its usage notes. A post-hoc sensitivity omitted the clinician score and refitted the four policies on identical records. Each policy selected the anchor on all three four-input folds. Substituting those outcomes lowered both continuous rules' overall safe savings to 6.18%. The Gain and vote results and all four breach totals were unchanged. [UCI documentation](https://archive.ics.uci.edu/dataset/161/mammographic+mass) · [Four-input results](../../thesis-paper/evidence/transport/birads-four-features.csv)

The [per-candidate scores](../../thesis-paper/evidence/transport/candidate-scores.csv), [task effects](../../thesis-paper/evidence/transport/task-contrasts.csv), [paired intervals and deletion counts](../../thesis-paper/evidence/transport/contrasts.csv) and [reference metric replay](../../thesis-paper/evidence/transport/reference-metric-audit.json) accompany the aggregate results.
