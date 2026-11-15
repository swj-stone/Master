# Masked predictions, Gain controls and alternative classifiers

The [frozen tenth reading](../../packages/transport-review-drafts/reading-10.pdf) and its standalone LaTeX sources are preserved in the [evidence archive](../../packages/thesis-robustness-evidence.zip). The evidence includes the records from the [earlier transport release](transport-release.md) and the following experiments. The earlier release remains available under its original checksum.

| Scientific question | Evidence and scope |
|---|---|
| Which refit metrics are attainable given an exact tree-ensemble transcript? | [Derivation](../ensemble-identification.tex), [enumerated depth-three fixtures](ensemble_cells/) and [formal lemmas](../formal/TranscriptCells.lean). The training data, models and admissible row sets belong to the specification. |
| What does each successive observation reveal on the finite outer dictionary? | [Observer definitions](../../experiments/context_review/transcript_observers.md), [704 paired-guard decisions](transcript_information/) and [individual transpositions](transcript_swaps/). The counts describe possible assignments of the stored vectors to the same ordered labels. |
| Can a label-free identifier also determine the decision? | [Protocol and checks](../../experiments/context_review/score_resolution.md), [precision ladder, five controls and unresolved cases](score_resolution/). Quantization uses the same binning at 2, 4 and 8 bits for each probability coordinate. |
| How much does voting add to Gain and a fixed tie rule? | [Matched nominee comparison](../../experiments/context_review/gain_increment.md) and [paired effects, costs and all folds](gain_increment/). The strict Gain winner is unique on the actionable slates. The broader comparison holds the candidate shortlist fixed. |
| What changes when the masking and receiving learners change? | [Crossed design](../../experiments/context_review/forest_cross.md) and [forest fits and rankings](forest_cross/) on ten regular and six wider tasks. Logistic, forest and boosted-tree receivers use their respective refitted anchors. |
| Does the result survive changing the anchor and candidate generator? | [ANOVA and exchange-neighborhood design](../../experiments/context_review/filter_neighborhood.md), [first draw](filter_neighborhood/), [three-draw protocol](../../experiments/context_review/neighborhood_replicates.md) and [within-task results](neighborhood_replicates/). The three draws reuse partitions, labels and learner seeds. |
| Does a continuous score gain from choosing between different vote counts? | [Paired tie and strict-order decomposition](ordinal_decomposition/). Each visited slate is held fixed. The comparison records changes in nominated and delivered supports, tolerance breaches and effective compression. |
| How fast is each rule with the same opportunity to stop? | [Timing and stopping protocol](../../configs/context_review/matched_stopping.json), [finished decisions, measured times and mask counts](matched_stopping/) and [proof of exact pruning](../formal/Challenger.lean). Each rule has a fresh cache. The timer covers nomination, including mask predictions. The archived timing measurements are inputs for recomputing the summary. |

Run the following from the thesis checkout. The verifier extracts the supplied archive into a fresh directory, regenerates its comparison tables from the archived probabilities and selection records, and checks an independent enumeration on small empirical orbits. It does not need the upstream feature matrices. Retraining the original learners requires the public data and the recorded study environments. Timings and prediction locks belong to the original runs.

```powershell
.\thesis-python.ps1 thesis-paper/package_revision.py --verify-transport packages/thesis-robustness-evidence.zip
```

The [artifact ledger](revision/artifacts.json) records the digest of the distributed archive. To build the latest supplement, follow the [decision release guide](decision-release.md). The archived robustness ZIP retains this edition. The verifier runs the replay code inside whichever archived edition is supplied. The [independent review history](../audit/robustness-review-history.md) identifies the manuscript editions read by the reviewers.
