# Independent audit of the downstream refit target

I used the source PMLB tables, the original fold receipts and public scikit-learn and XGBoost APIs. The scripts here do not import the runners being reviewed. I regenerated the stratified outer partitions, fitting and ranking rows, class encodings and source checksums. All comparisons use the original CPU benchmark learner and its fixed random seed. The results below represent an internal replication on the same computer and data.

## Which vetoes correspond to an observed loss after refitting?

There are 400 path proposals and 379 uniform-deletion proposals. Seventy-five occur in both sets. A unique proposal here means a specific subset within one particular fold. All 704 such proposals and 42 anchors were refitted using the outer development rows. Every saved outer probability was reproduced exactly. The failure label uses the refitted anchor as reference. At the specified 3% relative AUC and accuracy allowance, the observed test samples labeled 189 path proposals and 163 uniform-deletion proposals as failures. The reference models, candidate supports and holdouts recur across proposals.

The audit separately reconstructed candidate membership under the fixed random seeds, recomputed predictions under the frozen reference models for the ranking masks, generated the four weighting panels, and recalculated **48,720** recorded per-panel ballot values across nested, matched random, and separating families. This count includes repetitions when families share a mask. All ballots and the 8 derived rules' acceptances agreed. The effects below fix these ballots and vary only the test-sample definition of an excess loss.

| External loss allowance | Extra failures caught, path | Extra safe proposals vetoed, path | Extra failures caught, uniform | Extra safe proposals vetoed, uniform |
|---:|---:|---:|---:|---:|
| 1% | 14 | 17 | 3 | 6 |
| 3% | 11 | 20 | 0 | 9 |
| 5% | 5 | 26 | 0 | 9 |

Each entry compares the separating design with checking only the empty and complete restorations. “Safe” means the held-out empirical loss remained within the displayed cutoff. The actual tolerances in the vote stay unchanged. At 3%, seven of the eleven extra detections on the path proposals came from Sonar. Omitting that one dataset yields four additional detections and nineteen additional false vetoes. With equal weights for the datasets and equal weights for the folds within each dataset, the respective path changes were +2.20 and +3.97 percentage points per proposal. Both estimates are descriptive. The simple endpoint rule already vetoed 347/400 path candidates and 330/379 uniform-deletion candidates.

On the 400 path proposals, the extra vetoes reduce the cost $c_{FN} FN + c_{FP} FP$ only if a missed loss costs more than $20/11$ times a false veto. The equal-fold, equal-task counterpart is approximately 1.80. The 379 uniform-deletion proposals present no compensating extra detection at the original cutoff. A proposal-level error cost would need substantive justification. Training another model, keeping an expensive sensor and rejecting a redundant feature can have different practical costs. Aggregate error counts alone do not establish a useful selection policy.

Changing which deployed metric defines failure also matters. At 3%, an AUC-only target labels 128 path candidates as failures. The additional separating contexts catch seven and add 24 false vetoes. An accuracy-only target labels 158 failures, with six additional detections and 25 false vetoes. The two definitions yield the same change on uniform-deletion proposals, zero extra detections and nine extra false vetoes. [Sensitivity to the deployed target](refit-review-by-target.csv)

An additional diagnostic ranks proposals by four scalar scores derived without tuning a cutoff. A cheap score is the fraction of anchor features removed. A second uses the sum of the deleted features' ranks, with the strongest assigned weight $k$ and the weakest weight one, divided by the total rank weight. Two mask-based scores count the largest number of negative panels among the endpoints or among the entire separating family. Within a fold, ROC AUC asks whether a candidate above the 3% loss cutoff received a higher score than one below it. Folds with only one outcome have no estimable AUC. Each source has 28 eligible folds, covering 12 distinct path tasks or 11 uniform-deletion tasks. Fold AUCs were averaged within each available task, then tasks received equal weight.

| Score for a candidate | Path candidate AUC | Uniform-deletion candidate AUC |
|---|---:|---:|
| Fraction of anchor removed | 0.786 | 0.779 |
| Fraction of anchor rank weights removed | 0.739 | 0.747 |
| Weakest-context negative panel count | 0.639 | 0.641 |
| Endpoint negative panel count | 0.638 | 0.662 |

To separate size from identity, consider only pairs of candidates in the same fold with the same number of columns and discordant test-sample loss labels. There were 97 such pairs across 22 path folds and 137 pairs across 27 uniform-deletion folds. Pooling eligible pairs within each fold, then giving available folds and tasks equal weight, the separating score's concordance was 0.582 and 0.573. The endpoints gave 0.568 and 0.568. Size ties on every such pair. These exploratory ranks measure discrimination among proposals, not prediction accuracy of a final deployed model or a guarantee about subsets never offered to the selector.

## Direct Gain and the refit guard

The historical direct Gain rule uses every input column, breaks exact Gain ties by larger original index, evaluates every prefix width by refitting, and keeps the smallest width at the best ranking score up to the $10^{-14}$ tie convention. This differs from the paper's compact masked-prefix Gain anchor. I reproduced the gain order and all 1,209 width scores across 42 folds, independently recovered each selected width, and reproduced all 42 final prediction arrays exactly. The 1,251 independent fits include the full training-side rank model reused for the last prefix and the final development-side refit. Mean task AUC was 0.94328, accuracy 83.76% and retained fraction 74.48%. The original benchmark's 4.97 seconds/fold describe that run's local in-process timing; the audit did not use its own runtime to compare the selectors.

The refit guard reserves half of the original ranking rows as a separate adjudication sample, so comparisons with the unguarded original studies also change the sample available to the vote. The independently replayed partitions include 41 evaluable folds. Red Wine Quality fold 2 has just one ranking row from its rarest class (class counts 1, 9, 114, 107, 33, 3). That class cannot appear in both partitions. This failure was retained, and neither merging classes nor borrowing external rows was needed. For the 41 usable folds I recomputed compact Gain and both prefix controls, then refitted every distinct candidate on the inner and outer data (128 fits at each stage). The guard's decisions and all 287 method–fold records were reproduced, including every prediction, fallback and observed excess loss. The vote proposal itself in this guard audit is taken from the frozen proposal record.

In the 41 folds, an unguarded vote exceeded the refitted anchor tolerance 6 times, falling to 1 with the guard. Removing half of the Gain anchor's columns by rank exceeded it 29 times unguarded and 1 with the same guard. With equal task weights, the guarded vote reached AUC 0.93327 at 43.33% retained. The guarded prefix receiving the voting method's width reached 0.93327 at 43.13%, while the guarded independently sized half-anchor reached 0.93507 at 42.90%. The guard's value includes fallback to a larger support. The reductions in observed loss therefore offer weak evidence for a contribution specific to the voting step.

## Does the final minimum distinguish columns?

Forty of the 41 usable adjudication folds had at least two columns left at the original stopping support. One SPECTF fold had a singleton and no eligible one-column reduction. I exhaustively evaluated all 373 possible one-column deletions across the 40 folds, reconstructed every constant-weight separating family, and compared the maximizing support and weakest-Gain tie-break with the saved decisions. This required 3,181 distinct masked ranking predictions. All saved inner and outer outcomes agreed with independent refits (98 and 105 distinct models, respectively, counting fold-specific supports).

Among the 40 eligible folds, the best minimum was zero in 15, one in 13 and two in 12. In 32 folds several columns tied for the best score, including 15 where all candidate columns tied. The maximizing rule and the weakest-Gain deletion differed in 17 folds. Thus the score can differentiate candidate deletions on some folds, but its four-panel integer scale frequently leaves the choice to the Gain tie-break. The exact branch-and-bound result concerns computation of this finite statistic. It gives no ordering guarantee for the eventual refitted predictors.

At the same width the unguarded maximin challenger lost 0.00386 mean AUC and 0.29 percentage points of accuracy against dropping the weakest Gain column. After guarding, its mean AUC difference was −0.00044, accuracy difference +0.028 percentage points and retained fraction +0.47 percentage points. In the earlier guard experiment, the original vote improved unguarded same-width Gain by 0.00290 mean AUC and 1.28 percentage points accuracy before adjudication; after adjudication, the AUC gap fell to 0.000001. Each comparison pairs the models within a fold and equally weights the available task means. [All task-specific paired differences, including exact fold counts](refit-review-paired-means.csv)

## Scope of the PMLB rows

The local folds partition rows, and exact-vector grouping elsewhere in the replication only handles exact matches. The official PenDigits data come from 44 writers, with a separate 14-writer test set. The PMLB task redivides the aggregated feature table. Sonar contains echoes measured at multiple aspect angles, including 111 measurements from the metal cylinder. Its 208 feature rows cannot validate transfer to 208 distinct physical targets. These two source descriptions alone preclude treating all study rows as independent participants or objects. Other PMLB tasks in this audit have no independently reconstructed subject-, batch- or site-disjoint partitions. [Original PenDigits data description](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits) · [Original Sonar data description](https://doi.org/10.24432/C5T01Q)

## Forward gas-drift replication

I independently parsed the UCI archive, reconstructed the chronological splits, recomputed the compact Gain anchor, and reran native RFE at the two conditional widths and ANOVA at the two fixed widths. All selected columns matched the archive. I used the recorded vote and challenger supports to refit four guard and ten distinct final models. All per-row probabilities, the 14 method summaries, and the 42 held-out batch summaries matched. There were six observed classes in each future batch, with 294, 470 and 3,600 records respectively. The reported primary score weights each batch equally.

The full model's mean batch macro recall was 0.680, compact Gain's 0.617, the six-column voting support's 0.358 and the five-column maximin support's 0.346. Same-width five-column RFE reached 0.574. On batch 8 the maximin support missed every class-6 record. On batch 9 it missed every class-1 and class-6 record. Five-column RFE had positive recall in each of the 18 batch–class combinations. All three guarded arms reverted to 42-column compact Gain. This is one sensor-array series under drift. The three consecutive future batches share a training procedure and physical acquisition system. [Per-class counts and recalls](refit-review-drift-classes.csv)

## Reproduce the audit

The subsequent refit-on-prefix comparison reused the 41 eligible folds. A separate check reconstructed the candidate grid from the donor order, regenerated the stratified guard partitions and panel samples, and refitted every distinct candidate and selected support with the public model implementation. All 147 inner predictions and 63 final model predictions matched element by element. All 4,236 class and global panel decisions, every final width and each observed global and class-level loss indicator agreed. This check uses the same rows and labels. [Fold receipts](refit-review-grid.csv) · [Three decision rules](../../experiments/context_review/refit_vote_grid.md)

Run from the thesis checkout after creating the study outputs.

```powershell
.\thesis-python.ps1 thesis-paper/audit/refit-review.py target --fresh --ballots
.\thesis-python.ps1 thesis-paper/audit/refit-review.py gain
.\thesis-python.ps1 thesis-paper/audit/refit-review.py guard
.\thesis-python.ps1 thesis-paper/audit/refit-review.py challenger
.\thesis-python.ps1 thesis-paper/audit/refit-review-ranks.py
.\thesis-python.ps1 thesis-paper/audit/refit-review-drift.py
.\thesis-python.ps1 thesis-paper/audit/refit-review-grid.py
.\thesis-python.ps1 thesis-paper/audit/refit-review-pairs.py
```

[Threshold sensitivity](refit-review-sensitivity.csv) · [Task-specific changes](refit-review-by-task.csv) · [Independent candidate and fit counts](refit-review-target.csv) · [Independent Gain reconstruction](refit-review-gain.csv) · [Independent guard reconstruction](refit-review-guard.csv) · [Exhaustive challenger reconstruction](refit-review-challenger.csv) · [Within-fold ranks](refit-review-ranks-by-fold.csv) · [Fixed-width concordance](refit-review-ranks-same-width.csv)
