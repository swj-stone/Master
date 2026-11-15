# External tabular comparisons

The PMLB experiment uses the 14 tasks listed in [the design](../../configs/context_study/prospective.json). These are named measurement datasets selected before scoring. Multiple Features contributes only its Zernike representation. Red Wine Quality is a six-class task here. It is distinct from the cultivar classification data used earlier. The three stratified outer folds cover every row once. Each training fold has a separate fitting partition and a ranking partition. Results describe these tasks and tabular row splits. [PMLB v1.0](https://doi.org/10.1093/bioinformatics/btab727)

## Commands

From the thesis checkout, run the following commands in order. The acquisition step needs access to the pinned PMLB repository and its large-file storage. Subsequent commands use local inputs. The main runner and RFECV print progress and write a persistent log. The grouped sensitivity prints each current task and fold. Each worker has a one-hour limit.

```powershell
.\thesis-python.ps1 -m experiments.context_study.data
.\thesis-python.ps1 -m experiments.context_study.run
.\thesis-python.ps1 -m experiments.context_study.rfecv
.\thesis-python.ps1 -m experiments.context_study.grouped
.\thesis-python.ps1 -m experiments.context_study.antichain_matched
```

Run these additional regressions to compare optimized single-panel sampling and three original decision rules against the preceding implementation.

```powershell
.\thesis-python.ps1 -m pytest -q tests/test_context_study_panels.py
```

The primary output is `results/thesis_benchmark/prospective-pmlb-contexts-r1`. Its `sources.json` records the commit-specific large-file URL and digest for every compressed input and the upstream YAML metadata. Git LFS pointers are verified against the size and SHA-256 of the actual downloaded objects. An initial local acquisition attempt identified pointer files as compressed tables. The worker stopped at decompression before model fitting. The corrected download uses the same frozen commit and cohort. The local run retains the earlier pointer manifest and logs.

## What each arm measures

All primary arms finish with a new XGBClassifier on the selected features of the full development fold. Each uses 100 boosting rounds, depth 4, learning rate 0.1 and single-thread CPU histogram training. The fit does not use outer test labels. The saved selection file precedes test scoring. Its support indices refer to the original zero-based PMLB predictor columns.

| Arm | Implementation and source of its feature count | Training information and cost |
|---|---|---|
| Full | All predictors | Final training and prediction |
| Compact Gain | Existing XGBoost average split-gain ranking, restricted to model-used features. The original masked-prefix routine selects the smallest observed passing width in a search capped at 12 checks. | Reference fit on fit rows, sizing on ranking rows, fresh final fit |
| Matched Gain | First k columns of exactly the same Gain ranking, with k supplied by the nested Min arm | Includes the same reference fit and anchor computation. Reported time excludes the donor's search. It measures column identity at fixed width. |
| Supplementary antichain-matched Gain | First k columns of the frozen Gain ordering, with k supplied by the antichain Min arm on each of the 42 folds. The prefix is independently refitted on all outer development rows. | Conditional diagnostic designed after inspection of the primary results. The recorded duration covers final imputation, refit and prediction only. |
| L1 | scikit-learn OneVsRestClassifier with liblinear L1 logistic regression; use the union of nonzero coefficients. Examine four C values (0.01, 0.1, 1, 10). Select highest ranking AUC, then fewer retained columns, then smaller C. All-zero fits keep the largest fit-variance column to satisfy the common nonempty refit rule. | Standardization on fit rows, tuning on ranking rows, final XGBoost refit. The logistic model supplies only the support. |
| Point, empty majority, all point, pooled, nested Min | Common Gain anchor, cache and block/singleton search. Point uses one weighted panel and one context. Empty majority uses four panels and one context. All point uses one panel per distinct nested context. Pooled requires a majority of all ballots. Min requires a majority in every context. | The one-panel arms form and score only the original weighted panel. All include the reference fit and anchor. |
| Antichain Min and matched random contexts | Same nested Min learner and voting rule. Restore according to distinct constant-weight signatures or independent size-matched draws with rejection of duplicates. Include the same empty and full endpoints. | Same cached metric evaluations, early termination, final model and accounting as nested Min. The input mask budgets match before short-circuiting. |
| Supplementary RFECV | Native scikit-learn RFECV selects its width by 3-fold inner CV over the outer development fold. Recursive elimination drops 20% of original width per step. A scikit-learn pipeline refits the median imputer and the same XGBoost estimator inside each training fold. Importance comes from the fitted estimator's Gain. Binary scoring uses ROC AUC, multiclass scoring uses macro one-versus-rest AUC. | Includes ordinary RFECV fitting, all candidate widths, its final estimator, plus a fresh XGBoost refit and test prediction under the same evaluation rule. The separate timing campaign can face different processor load. |

The exact settings, iteration order and fallback rules are in [benchmark.py](benchmark.py), [selection.py](selection.py), [rfecv.py](rfecv.py) and the [RFECV protocol](../../configs/context_study/rfecv.json). Where possible, the primary runner computes common reference fits and anchor sizing once per fold and charges the measured cost in full to each arm that needs it. Each arm refits and predicts separately even when two supports agree. Primary times exclude disk I/O and worker startup. These times cannot be compared directly with the earlier cold-process benchmark table.

## Inputs, grouping, and reuse

PMLB distributes standardized historical tables, some of which contain repeated measurements or identical feature vectors. Dataset-specific licensing fields were absent from the 14 pinned YAML records. Upstream source fields were populated for Multiple Features, Libras Movement and Wine Quality. Software repository terms and upstream dataset provenance are recorded separately. Reproduction downloads measurements directly from the pinned source. These materials support a retrospective methodology comparison with no clinical-use inference. [Pinned PMLB repository](https://github.com/EpistasisLab/pmlb/tree/7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68) · [PMLB software license](https://github.com/EpistasisLab/pmlb/blob/7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68/LICENSE)

A secondary audit marks test rows for which the full untransformed feature vector also occurs in that fold's development set. It recomputes the frozen methods' test scores after removing marked rows. The grouped refit runs every dataset with a marked fraction above 10%, using feature-vector-disjoint outer folds and feature-vector-disjoint fit/ranking partitions. Duplicate vectors with conflicting outcomes stay together. This grouping cannot reconstruct unavailable subject, time, or site identifiers. [Audit design](../../configs/context_study/duplicate-audit.md) · [Grouped design](../../configs/context_study/grouped.json) · [Grouped runner](grouped.py)

RFECV and grouped results are in `results/thesis_benchmark/prospective-pmlb-rfecv-r1` and `results/thesis_benchmark/prospective-pmlb-duplicate-groups-r1`, respectively. Each arm has a fold-level record, row-aligned probabilities, frozen supports, and source or protocol receipts. RFECV retains the complete inner validation curve. The grouped study additionally records the outcome conflict counts and associated test sensitivity.

The underlying estimator and feature-selection interfaces are documented by their maintainers. Package versions used in this experiment are recorded in the primary protocol lock. [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) · [scikit-learn logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) · [scikit-learn RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)

The antichain-matched Gain supplement uses [its own fixed design](../../configs/context_study/antichain-matched.json) and [refit entry point](antichain_matched.py). The fold table is in `results/thesis_benchmark/prospective-pmlb-antichain-matched-r1/per-fold.csv`.

Run `.\thesis-python.ps1 -m experiments.context_study.package` to recompute all 630 sets of metrics from the saved row predictions and verify input lengths, selected-column identities and recording hashes. The local [validation evidence ZIP](../../packages/thesis-pmlb-evidence-r1.zip) contains row IDs, encoded labels, probabilities and receipts, alongside the compact tables and study protocols. Original feature measurements can be obtained from PMLB with the acquisition command above.

## Conditional compression and restoration controls

The later [same-width RFE and ANOVA design](../../configs/context_study/matched_selectors.json) gives two established selectors the size chosen by the separating design in each fold. Native scikit-learn RFE repeatedly fits the tree learner on the development sample, eliminates by split gain in steps of 20% of the original width, and stops at the supplied width. Native SelectKBest ranks the original columns by their one-way ANOVA F statistic. Both see all original columns and all development rows. Every support is saved before evaluation with the common refitted predictor. These controls measure quality at equal compression. Their own measured time excludes width selection. [Independent reconstruction](../../thesis-paper/audit/rounds/matched_selectors_replay.py)

Five further [random families](../../configs/context_study/random_families.json) per fold measure between-draw variability at fixed restoration counts and sizes. In the [fixed-margin study](../../configs/context_study/fixed_margins.json), random reassignment of codewords retains full pair coverage, and bipartite switches retain each mask size and each feature's appearance count. Each of the latter two comparisons has three seeds per original fold. All reuse the original outer folds. The [complete study instructions](../../thesis-paper/README.md) list the runnable commands.
