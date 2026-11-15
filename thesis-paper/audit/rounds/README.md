# Independent comparator and prediction audit

The checks below use the saved raw tables, split and support receipts, public scikit-learn and XGBoost APIs, and the shared Python environment. The replayers do not import the experimental selection or evaluation code. They reconstruct predictions for each archived support. For the controls identified in the third column they also reconstruct the support from its inputs. The two matched-width controls take the donor's observed width as a condition of the comparison. All prediction comparisons gave a maximum absolute difference of zero on this machine.

| Study | Evaluations with a fresh final refit | Supports independently reconstructed | Additional checks |
|---|---:|---:|---|
| PMLB primary, 14 datasets × 3 folds × 11 arms | 462 | 168 | Full, compact Gain, nested-width Gain, tuned L1, class order, source hashes and split replay |
| PMLB RFECV, 14 × 3 | 42 | 42 | Each width and all three inner validation scores, selected features, averaged AUC and held-out predictions |
| Exact-feature-vector grouped PMLB, 4 × 3 × 7 | 84 | 36 | Full, compact Gain and tuned L1. Outer and inner partition seeds, row sets and group separation |
| PMLB Gain at separating width, 14 × 3 | 42 | 42 | Donor identity and length, ranked prefix, separate final refit |
| NIPS challenge, 3 × 5 × 9 | 135 | 75 | Unrestricted full, screened full, coefficient anchor, separating-width coefficient prefix and tuned L1 |
| **Total** | **765** | **363** | Each stored test score matched its probabilities and labels |

Repeated identical PMLB supports on the same primary fold can share an audit refit. Each method's recorded predictions are checked independently. The support column excludes vote-derived supports. Replaying those search decisions is a separate task. In the six older development datasets a further 270 historical records passed the source, support-rule, frozen-selection and metric checks described below. These older checks use the archived predictions without refitting every older method.

## Reproduce

Two additional width-matched native selectors were added after this 765-result audit. Independent reconstruction of their 84 supports and predictions reproduced every saved probability, bringing the corresponding totals to 849 prediction checks and 447 reconstructed baseline supports. RFE's support is reconstructed by a manual recursive elimination loop; ANOVA's support is reconstructed from its F statistics and the library's stable ordering. At the separating selector's mean retained fraction of 37.89%, AUCs were 0.92808, 0.92872 and 0.91892 for separating, RFE and ANOVA, respectively. [Fold records](matched-selectors-replay.csv) · [Reproducer](matched_selectors_replay.py)

Run from the thesis checkout after the corresponding result folders and original input tables have been created.

```powershell
.\thesis-python.ps1 thesis-paper/audit/rounds/pmlb_baselines.py
.\thesis-python.ps1 thesis-paper/audit/rounds/nips_baselines.py
.\thesis-python.ps1 thesis-paper/audit/rounds/supplemental_baselines.py rfecv
.\thesis-python.ps1 thesis-paper/audit/rounds/supplemental_baselines.py grouped
.\thesis-python.ps1 thesis-paper/audit/rounds/supplemental_baselines.py antichain_matched
.\thesis-python.ps1 thesis-paper/audit/rounds/historical_controls.py
.\thesis-python.ps1 thesis-paper/audit/rounds/matched_selectors_replay.py
.\thesis-python.ps1 thesis-paper/audit/rounds/expected_coverage.py
.\thesis-python.ps1 thesis-paper/audit/rounds/random_family_replay.py
.\thesis-python.ps1 thesis-paper/audit/rounds/random_family_replay.py --fixed-margins
.\thesis-python.ps1 thesis-paper/audit/rounds/replay_revision.py
```

The grouped audit uses the duplicate counts from the primary replay to redetermine the follow-up cohort. The scripts verify source and data receipts and write the accompanying CSVs. Run within the pinned thesis environment. Cross-version bitwise identity is a separate question.

## PMLB controls and interpretation

**Cost at the supplied width.** In the follow-up RFE and ANOVA jobs, mean times for imputation, selection, and the shared-protocol final refit/prediction sum to 1.400 s and 0.128 s per fold, respectively. RFE spends 1.274 s in the recursive selector and ANOVA 0.00114 s in the univariate selector. These times assume the width is already known. Both controls use all outer development rows. They were run in a separate timing batch. The separating-width Gain supplement records 0.295 s for the final refit and prediction on its supplied prefix. It consumes an already fitted ranking.

**Compact Gain.** Fit a 100-round, depth-four XGBoost model on the fitting partition. Rank split-used features by decreasing average split gain, breaking ties by original column position. Total split gains give a starting width via the squared sum divided by the sum of squares. A frozen-model search probes up to 12 masked prefixes on all ranking rows. A prefix must retain at least 92% of full-ranking AUC and accuracy, plus average precision in the rare binary case. The widest split-used prefix has the same predictions as the full model. The algorithm returns the smallest *observed* passing width. A nonmonotone performance curve could hide another passing width. Missing columns take medians of up to 512 fitting rows. The later deletion vote uses up to 256 ranking rows.

**Donated widths.** In the primary suite the matched prefix receives the nested selector's width. The follow-up prefix separately receives the separating selector's width. At the latter width the separating selector has mean task AUC 0.92808 and the Gain prefix 0.92237. The per-task difference is positive on nine, negative on two and zero on three. A conditional prefix's timing omits the cost of discovering its width. Comparing independently chosen widths favors less compressed predictors on these data. Compact Gain achieves 0.9374 AUC with 47.0% of the original columns, compared with 0.9281 with 37.9% for separating.

**L1 support.** Fit the imputer and standardization on the fitting partition. Fit one-vs-rest liblinear L1 logistic models for four values of C (0.01, 0.1, 1 and 10). Choose by ranking AUC of each fitted model's probabilities, breaking score ties by smaller support and then smaller C. Take the union of nonzero coefficients across classes. The implementation uses the greatest fitting-set variance feature when a candidate's coefficient union is empty. Imputation, scaling, all candidate fits and tuning use only the outer development sample. The published outcome is from a fresh common tree predictor on the resulting columns.

**RFECV.** The wrapper runs three-fold stratified internal CV over the entire outer development sample. It uses the same XGBoost parameterization as the other final predictors, fits a median imputer within each inner training fold, and removes a block of 20% of the original width per step, rounded down with a minimum of one. AUC selects the width. Independently running scikit-learn's RFECV reproduced all 42 supports and the complete per-inner-fold validation curves. RFECV's ordinary fitting procedure includes a final estimator. The benchmark performs another common fresh refit and evaluation to match its final-prediction procedure. Mean recorded time was 3.221 s for RFECV fitting plus 0.222 s for the additional refit/evaluation, versus 0.876 s total for separating. Removing the additional 0.222 s would leave the broad timing ordering intact. RFECV had higher mean AUC (0.9457) and retained more columns (79.5%). [scikit-learn RFECV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)

**Timing and statistical units.** The PMLB workers attribute shared fitting and anchor preparation to each route that requires them and include final imputation, fitting and prediction. Downloading, process startup, serialization and writing predictions are outside these timers. Timings are local observations using one thread. RFECV and the separating-width control were follow-ups in separate batches. Averaging folds within each dataset and weighting the 14 datasets equally describes the measured tasks. Shared observations across folds preclude counting 42 independent studies. Similarly, averaging AUCs over folds is a different operation from computing one AUC on concatenated out-of-fold scores.

## Repeated input values and validation boundaries

The primary split gives each row exactly one out-of-fold prediction. It can place identical complete input vectors on opposite sides of a split. Raw row-vector groups exclude the target and precede imputation. The verifier checks hash-group and direct feature-vector agreement. Across the 14 datasets 1,003 test rows have a development-side duplicate. Four datasets meet the follow-up selection rule of more than 10% overlap.

| Dataset | Overlapping test rows / original rows | Proportion | Distinct feature vectors |
|---|---:|---:|---:|
| Movement Libras | 44 / 360 | 12.2% | 330 |
| Segmentation | 314 / 2,310 | 13.6% | 2,086 |
| SPECTF | 106 / 349 | 30.4% | 267 |
| Wine Quality Red | 319 / 1,599 | 20.0% | 1,359 |

Removing test rows whose feature vector also occurred in development, while keeping the trained primary models fixed, changes the separating-minus-nested task-average AUC from +0.002460 to +0.001721 and separating-minus-random from −0.000529 to −0.001108. All classes remain represented in the restricted fold scores. A second sensitivity study reruns the whole pipeline on the four affected datasets. It keeps identical feature vectors together across training, ranking and test. The four-dataset AUC means were 0.87915, 0.87046 and 0.86936 for separating, matched random and nested. Their respective accuracies were 73.12%, 73.77% and 74.21%. Higher AUC and higher hard-decision accuracy occur under different context designs in this sample. Dataset identifiers, feature matches and label stratification alone cannot establish independence of patients, sessions or acquisition times.

## NIPS feature-selection challenge

The audit independently decodes dense Arcene, sparse term counts on Dexter and sparse binary indices on Dorothea. The latter two are read with scikit-learn's LIBSVM parser after adding dummy row labels and making the Dorothea unit entries explicit. Archive and member SHA-256 values agree. Each split uses the official training labels, stratified 70:30, for fitting and ranking. A MaxAbs scaler and univariate F statistics use fitting rows only. A stable sort retains the top 256 columns with a defined F statistic, with original column order breaking ties. All subsequent screened selectors share these 256 columns. The full raw L2 control uses the original 10,000, 20,000 or 100,000 columns. Its final fit uses the complete official training set on each repetition and its BER and AUC were identical across the five seeds for each dataset.

The anchor ranks the 256 screened features by decreasing absolute L2 logistic coefficient after MaxAbs scaling. A direct median-masking replay reproduced the first ranking prefix meeting the 8% AUC and accuracy tolerance. Voting then further reduces this anchor. The width-matched prefix receives the separating selector's width. Tuning L1 here scores each nonempty candidate support by the ranking AUC of a fresh L2 logistic model, unlike the PMLB L1 procedure. The all-empty fallback was never needed. For voting and screened controls the original code stores sorted raw input IDs, while the matrix passed to the final solver retains selection ordering. The replayer recovers this ordering and reproduced all 135 official-validation predictions exactly.

The public labeled competition **validation** cohort is scored only after all supports were frozen for that dataset. The five training/ranking seeds reuse its outcomes. No hidden competition test labels enter the comparison. The source archives give independent datasets, while each dataset contributes just one external validation sample. A conditional matched-width advantage is largest on Arcene and Dorothea. Screened-full logistic attained lower BER than separating on all three (24.7/9.1/26.0% versus 29.7/17.4/32.4%). The 256-column supervised screen accounts for most of the raw-width reduction. Timings record the components for screening, anchor, conditional selection and final fit. The NIPS driver checks its one-hour budget between completed arms, so a single blocked solver call could overrun that check. All measured arms completed.

## Historical methods on the six development datasets

The retained protocol resolves eight baseline implementations plus a full reference on each of 30 splits. The audit hashes the actual benchmark entry point and baseline sources, checks selected columns against stored rankings or intrinsic decision traces, verifies the signed selection manifest and held-out data identity, and recomputes final metrics. It also reconstructs every ShapG correlation graph's node degrees from the training matrices using a global sorted-edge implementation. This checks which configurable algorithms actually ran. Recomputing every historical coalition refit is outside the replay.

| Archived label | Calculated quantity or decision | Scope of comparison |
|---|---|---|
| TreeSHAP | XGBoost real-valued raw-margin contributions with exact contribution mode, bias excluded. Mean absolute value over explained training rows (up to 1,024) and model outputs. | Attribution supplies a ranking. The study's validation-prefix search chooses width. [XGBoost prediction documentation](https://xgboost.readthedocs.io/en/stable/prediction.html) |
| SamplingSHAP | The installed SHAP SamplingExplainer performs paired sampling on frozen predictor outputs. Up to 32 ranking observations and 128 background training observations, with the argument nsamples set to 12 times input width per explanation. Absolute local scores are averaged across observations and outputs. | The actual hybrid prediction counts differ from the formal budget argument and are recorded. Common prefix selection follows attribution. [SamplingExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.SamplingExplainer.html) |
| ShapG | Global absolute-Pearson edge order with first incident edges covering all training features, one-hop graph, at most three sampled neighbors per draw, normalized local subset contributions. Coalition models refit with classification binary/macro F1 or regression ordinary R². | Covered-node construction reproduces the published Algorithm 1 pseudocode. Reconstructed graphs had 1–29 components and 26/30 were disconnected. A requirement in the surrounding prose that the graph be connected would describe a different implementation. Common prefix sizing chooses widths. [Zhao, Liu and Parilina, Algorithms 1 and 3](https://arxiv.org/html/2407.00506v2) |
| MinShap | Minimum sampled loss reduction over four permutation paths, per-row paired loss variance and the alpha=0.05 threshold. Observing a negative marginal rules out selection for that feature in the specified sample. | The finite-path test returned 14 empty supports including five Fraud splits. Constant predictions were evaluated. The other empty-support counts were three each on Abalone, Madelon and ISOLET. The original per-run regression utility clips negative R² to zero. The published per-seed table and audit report ordinary test R² separately. [Zheng and Raskutti](https://arxiv.org/abs/2604.15107) |
| Banzhaf | Signed marginal contrast from 32 uniform complementary mask pairs, 64 sampled mask observations, native subset-refit utility. | Finite sampled ranking plus common prefix sizing. |
| Permutation importance | Three random shuffles per split-used column on up to 128 ranking rows, positive mean utility drops retained, highest-score fallback if needed. | Finite-sample permutation estimate and an explicit decision threshold. |
| NSGA-II | Population eight plus two offspring generations, 24 logical candidate evaluations, maximizing ranking utility and minimizing cardinality. | One representative Pareto point chosen by utility, then smaller size and original feature order. Caching can reduce the actual distinct fits. |
| Compact Gain / Full | Average split gain with the frozen validation-prefix rule, or all columns. | Both use the same historical prepared inputs. |

Published papers do not report these locally measured numbers. The datasets, learners, attribution targets, budgets and conversion of scores into supports determine what each row means. The old cold-process timings and new in-process experiments have different boundaries. The older tasks informed method development, so their metrics supply descriptive history.

## Provenance and reuse

PMLB source revision 7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68 supplies a Git LFS pointer and associated compressed table for each dataset. The replay checks the pointer's byte size and SHA-256 against each archive and checks the dataset metadata manifests. PMLB repository licensing and metadata do not establish uniform reuse terms for all underlying collections. Use its pinned identifiers, source references and acquisition instructions for redistribution decisions. [PMLB repository and license](https://github.com/EpistasisLab/pmlb/blob/master/LICENSE)

The challenge data come directly from UCI. Each of the three dataset pages identifies its dataset-specific CC BY 4.0 terms. Preserve these attributions in any distribution that includes the underlying observations. [Arcene](https://archive.ics.uci.edu/dataset/167/arcene), [Dexter](https://archive.ics.uci.edu/dataset/168/dexter), [Dorothea](https://archive.ics.uci.edu/dataset/169/dorothea)
