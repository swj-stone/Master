# Voting directly on class recall after refitting

This retrospective experiment asks whether class-specific votes identify smaller Gain prefixes that survive a new model fit. It reuses the original 41 eligible refit-guard folds and their compact Gain anchors. Red Wine fold 2 remains ineligible because one class had a single ranking row. The choices below were written to [the locked design](../../configs/context_review/refit_vote_grid.json) before scoring the newly selected prefixes. The outer folds had already been examined in earlier experiments.

For an anchor with k columns, take the prefixes ending at one quarter, one half, three quarters and all of k. Round fractional widths up. Collapse duplicated widths. Fit the same XGBoost model separately on each prefix, using the original fitting and proposal rows. All three selection procedures use the same predictions on the untouched adjudication rows. The global point guard selects the smallest prefix within 3% relative AUC and accuracy loss from the refitted anchor; it also checks relative average precision at 5% when the original rare-binary condition applies. The class point guard adds the requirement that recall for each class remain within 3% relative loss.

The four-panel rule uses the original adjudication rows plus three paired class-stratified bootstrap replicates of those same rows. Define one voting context for the global guard and one for recall of each class. In each context a panel votes yes when the candidate meets the corresponding tolerance against the anchor evaluated on that panel. Select the smallest prefix for which the minimum yes count across contexts is at least three. Thus every context can tolerate one dissenting panel, and the dissenting panel may differ by context. Each fold shares its four panel draws across all widths. The anchor is always eligible. A class for which the anchor gets no rows right imposes no restriction. A class with a single guard example has identical class-recall votes on all four panels.

The earlier context-deletion ballot used masked predictions of a single fitted model. This experiment trains a model on each candidate prefix and defines the minimum across global and class adjudication contexts. It is a different algorithm using the same voting and minimum principles. Comparisons among the three rules share candidate identities, fits, split and learner. The benchmark charges the common preparation, refits, construction of the complete ballot table and outer development refits. Its logical short-circuit count measures panel-context evaluations that would suffice to decide the examined widths. Here the full table is actually computed for cross-checking. Saved evaluation work in the logical traversal therefore does not establish training speed or wall-clock acceleration.

## Outcomes on the archived study

The table gives equal weight to the 14 datasets, averaging the available folds within each dataset first. The three rightmost columns count folds out of 41. A class breach occurs if any class's external recall loses more than 3% of the refitted anchor's recall. No fold qualified for the inherited rare-binary average-precision check.

| Rule | Outer AUC | Accuracy | Macro recall | Input columns retained | Folds smaller than anchor | Global breaches | Class breaches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compact anchor | .93487 | .82256 | .76725 | 48.54% | 0 | 0 | 0 |
| Global point guard | .93054 | .82018 | .76521 | 38.63% | 20 | 2 | 16 |
| Global and class point guard | .93139 | .82119 | .76645 | 43.21% | 11 | 2 | 8 |
| Four-panel minimum vote | .93089 | .82118 | .76496 | 44.26% | 10 | 3 | 7 |

Adding class recall to the point guard coincided with eight fewer fold-level class breaches, at an average cost of 4.58 percentage points of input columns retained. Replacing the class point estimate with four panels changed 4 of the 41 selected widths. It rejected three of 14 previously passing non-anchor candidates and accepted one of 92 previously rejected candidates. On New Thyroid fold 0 the newly accepted two-column candidate reduced macro recall from .911 to .850 and crossed the global loss tolerance. On Satimage fold 1 and Vehicle fold 0 the vote returned to the anchor; on SPECTF fold 2 it chose 17 columns in place of six. These observed choices account for the entire difference between the two class-aware rules. The distribution of scarce adjudication cases also constrains the comparison. Five folds contained a class with a single guard row; twelve contained a class with at most two. The three bootstrap panels add no class-recall information for singleton classes. Results excluding the partial Red Wine task are available in [the 13-complete-task summary](../../results/context_review/refit-vote-grid/complete-tasks.csv).

The full run used 147 shared adjudication models and 63 deployment models across the 41 folds. Adjudication fitting and prediction took 48.14 s in total, deployment fitting and prediction 9.05 s, and construction of all ballots 1.53 s. Measured worker time summed to 62.00 s, and no fold exceeded 3.47 s inside its worker. The sequential driver finished in about 160 s with process starts. These timings exclude the cost of finding the inherited anchor. The four-panel procedure requires 576 of 2,960 possible logical context-panel checks over widths visited before the decision. The single-panel class rule requires 199 of 727 and the global point rule 93 of 93. Short-circuiting prunes most of the four-panel table, while the direct comparison still charges more checks than either point rule.

The retrospective results support a descriptive tradeoff for class constraints. Four panels produced one fewer observed class breach than one panel while retaining an additional 1.05 percentage points of input columns, increasing global breaches from two to three and lowering task-mean AUC and macro recall. These data give no evidence of a mean prediction or compute advantage from the extra ballots.

## Reproduction

Run from the thesis checkout after the archived refit-guard study is available.

```powershell
.\thesis-python.ps1 -m experiments.context_review.refit_vote_grid
.\thesis-python.ps1 -m experiments.context_review.refit_vote_grid_report
```

The runner records elapsed and estimated remaining time after each fold and stops a worker exceeding one hour. Guard rows, panel memberships and all prefix probabilities are saved in each fold's `guard.npz`. The fold's `selection.json` is written and hashed before fitting the deployment models or scoring outer labels. Frozen choices, complete votes, counts and predictions are available under [the run directory](../../results/context_review/refit-vote-grid/). [Per-fold choices and scores](../../results/context_review/refit-vote-grid/per-fold.csv) · [Candidate ballots and decisions](../../results/context_review/refit-vote-grid/candidate-decisions.csv) · [Paired task differences](../../results/context_review/refit-vote-grid/paired-differences.csv) · [Shared computational cost](../../results/context_review/refit-vote-grid/costs.csv)
