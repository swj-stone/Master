# Continuous masked predictions and refit ambiguity

The historical data contain 704 distinct proposed supports on 42 folds from 14 tasks. Comparing *different proposals* at the same width on the same fold gives zero conflicting groups with identical recorded continuous metrics. Thresholding those metrics gives 31 conflicting ballot groups containing 136 proposals. These counts describe duplicates in the recorded sample. Each proposed support has a different counterfactual question: can its own refit pass in one world and fail in another while the stated observations agree?

For each support we fixed every development row and label, the fitted initial classifier, the refitted candidate and anchor models, the ordered outer labels and the empirical multiset of complete outer feature vectors. We reassigned those vectors to outer label positions only within sets of rows with identical scores for every supplied query. Every swap preserves the joint distribution of the feature columns themselves. It can change the association between those columns and the target. Models and selected supports are fixed throughout. Relative tolerance is 3% on both AUC and accuracy.

| Information available on top of the original proposal observations | Supports with opposing tolerance decisions on the allowed permutations | Invariant decisions |
|---|---:|---:|
| No outer predictions | 704 | 0 |
| Outer frozen predictor on the candidate and anchor with the tail hidden | 17 | 687 |
| Outer unmasked frozen predictor alone | 0 | 704 |
| Same two outer queries plus the unmasked frozen predictor | 0 | 704 |
| Outer frozen predictor on the candidate and anchor for every recorded restoration | 0 | 704 |

The 17 include ten originals that passed and seven that failed. They come from Ionosphere (3), Segmentation (1), SPECTF (12) and Vehicle (1). The anchor with its entire tail restored gives the unmasked frozen prediction, so interior restorations resolved no additional tolerance decisions once that endpoint was observed. The additional outer predictions are diagnostic measurements. The deployed selector ordinarily sees the proposal sample. The unmasked probability vector alone uniquely labeled 35,619 of 38,021 outer rows within their respective folds. The [precision controls](../../thesis-paper/evidence/score_resolution/summary.csv) compare ordinary probabilities to five recodings with the same code frequencies.

If the perturbation is limited to one transposition between oppositely labelled outer rows, 191 of the 704 proposal-only decisions can reverse. With both empty-restoration queries visible, only 234 proposals admit such an exchange and ten have a reversal. Those ten give 60 reversing pairs among 7,774 eligible pairs across all proposals. Counting possible exchanges assigns no probability to changes in future samples. [All transpositions by proposal and task](../../thesis-paper/evidence/transcript_swaps/)

For each permutation both models' metrics are recomputed on the same assigned rows. A pass requires both candidate-minus-0.97-times-anchor gaps to be nonnegative up to the original numerical slack. Under a fixed multiset, average ranks of each class's predicted probabilities are fixed. Binary or macro one-versus-rest AUC and accuracy gaps are affine in the assignment of each feature row to a label. A linear assignment yields the actual extrema of each gap. We first searched smaller blocks for explicit counterexamples; the remaining 26 cases were revisited without that restriction. For these records a witness or a single-coordinate bound resolved every decision. In general two separately attainable coordinate maxima need not come from the same assignment. The code reports any unresolved case.

Giving the observer every recorded outer mask, the unmasked predictor and the final reference anchor's output at every outer position creates a stronger test. In this test at least one individual reduced-model score can still change on 93 supports. Only three AUCs vary (all Phoneme, fold 1), no accuracy changes and all tolerance decisions are invariant. A change in row scores within a single outcome class can leave aggregate metrics unchanged. Across other admissible input domains the sharp tree-ensemble set can be wider.

The archive contains per-row equality classes for the frozen model's unrounded, implementation-precision probability vectors, labels, anchored and reduced refit predictions, the source and split hashes, and index maps for the counterexamples. Rebuilding those equality classes from the original features reproduced all stored continuous proposal metrics before any outer permutation was tested. Use the shared environment in the thesis checkout to replay the evidence.

```powershell
.\thesis-python.ps1 -m experiments.context_review.transcript_permutations replay
.\thesis-python.ps1 -m experiments.context_review.transcript_information
.\thesis-python.ps1 -m experiments.context_review.transcript_swaps
```

With the original pinned PMLB feature files available, `transcript_permutations freeze` regenerates the classifier predictions and row classes. [Observer-by-proposal outcomes](../../thesis-paper/evidence/transcript_information/proposals.csv) · [Permutations and paired metrics](../../thesis-paper/evidence/transcript_information/witnesses.json) · [Fixed-anchor audit](../../thesis-paper/evidence/transcript_permutations/summary.csv) · [Ensemble characterization](../../thesis-paper/ensemble-identification.tex)
