# Ranking passing candidates within a fold

One nomination followed by a refit guard measures the outcome of the winning candidate. This retrospective analysis uses every evaluated half-width candidate to measure how well the frozen nomination scores order a passing deletion above a failing one. The two targets are passage on the outer fold and passage on both the guard and outer fold, evaluated against the corresponding receiver's anchor. The original 3% relative AUC and accuracy tolerances and the 5% average-precision tolerance on flagged rare folds apply with the original numerical slack of 1e-12.

Within each fold, compare every positive candidate with every negative candidate, assign a win one point and an exact score tie half a point, and divide by the number of pairs. We use the unrounded stored median-mask scores and integer ballots without a tie tolerance. This equals ROC AUC among the candidates of the fold. The candidate pairings share supports and test observations. The table averages informative folds within a source, then weighs the contributing sources equally. An uninformative fold has either no candidates or a single observed outcome class. The first panel contains 30 folds (20 with candidates) on ten sources, and the wide panel contains 16 folds (15 with candidates) on six.

| Panel and receiver | Passing target | Source count | Mixed folds | Candidate pairs | Point | Full margin | Vote | Gain order |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ten, XGBoost | Outer | 5 | 10 | 118 | .752 | .717 | .540 | .491 |
| Ten, XGBoost | Joint | 5 | 9 | 100 | .674 | .634 | .550 | .405 |
| Wide, XGBoost | Outer | 5 | 12 | 128 | .630 | .621 | .500 | .630 |
| Wide, XGBoost | Joint | 4 | 8 | 93 | .604 | .598 | .521 | .612 |
| Ten, logistic | Outer | 5 | 9 | 69 | .607 | .614 | .514 | .617 |
| Ten, logistic | Joint | 4 | 7 | 75 | .566 | .534 | .542 | .415 |
| Wide, logistic | Outer | 5 | 9 | 106 | .643 | .636 | .562 | .571 |
| Wide, logistic | Joint | 5 | 8 | 102 | .500 | .498 | .546 | .590 |

The Gain policy's operational tie order first sums the original within-anchor column ranks, then orders equal sums lexicographically by sorted column indices. The archive contains both the full order and the rank sum with equal sums tied. A higher exported score denotes a better rank. Feature selection and nomination use the same tree for both receivers. Retaining the guard in the target changes the positive class and can change which folds contribute.

On the wider XGBoost outer target, the point-versus-vote difference is .130 (paired task bootstrap interval [.105, .155]). Per-source concordance of the point score ranges from .458 on SECOM to .896 on Ozone. The Gain order averages .630, closely matching the point score. Switching to logistic joint passage changes the point-versus-vote contrast to −.046 [−.235, .093]. The intervals resample contributing sources 30,000 times with replacement and condition on their folds, candidates and outcomes. The number of candidate pairs describes the comparisons entering the statistic.

Run `./thesis-python.ps1 -m experiments.context_review.candidate_concordance` from the thesis checkout. [Candidate rows](../../thesis-paper/evidence/concordance/candidates.csv) join the frozen scores to both pass decisions by panel, dataset, fold, receiver and support. [Fold results](../../thesis-paper/evidence/concordance/folds.csv) and [source results](../../thesis-paper/evidence/concordance/tasks.csv) include rows with undefined concordance. [Summary](../../thesis-paper/evidence/concordance/summary.csv) and [paired contrasts](../../thesis-paper/evidence/concordance/contrasts.csv) give the reported estimates.
