# Does the separating vote add information to deletion size?

A model that used both the deletion fraction and the vote ranked failing proposals above passing ones more often than a model using size alone on held-out tasks. The gains in candidate-ranking AUC were 0.024 on search paths and 0.046 on separately drawn proposals. The primary measure, log loss, rose by 0.002 on paths and fell by 0.013 on drawn proposals. Both paired log-loss intervals include zero.

## Proposals and prediction

The outcome is whether the candidate exceeded the original allowed loss against a freshly refitted anchor on the same outer rows. The allowance was 3% of anchor AUC or accuracy, and 5% of average precision where applicable. Each model was fitted on 13 tasks and used to predict every proposal on the 14th, cycling through all tasks. Search-path and drawn proposals had separate training and evaluation loops. Of 400 path and 379 drawn records, 75 fold–support pairs occur in both collections, giving 704 unique pairs. The earlier study had already examined the outcomes of these tasks. The present model specifications and analysis were locked before their new held-out predictions were scored. [Analysis design](../../configs/context_review/incremental_prediction.json) · [Input and code hashes](../../results/context_review/incremental-prediction/protocol.json)

Both logistic models used the fraction of anchor columns removed and three piecewise-linear terms with fixed knots at 0.25, 0.50 and 0.75. The augmented model added the largest fraction of dissenting panels among the separating restorations. Training used equal weight per task, per available fold within task, and per proposal within fold. Each predictor was standardized using the resulting training weights. The objective was mean log loss plus 0.005 times the sum of squared standardized coefficients, with an unpenalized intercept. Held-out task outcomes were used only for evaluation. [Fitted coefficients and training task identities](../../results/context_review/incremental-prediction/models.json) · [All held-out predictions](../../results/context_review/incremental-prediction/predictions.csv)

## Paired out-of-task results

Differences below are size-plus-vote minus size. Smaller log loss and Brier are better. Larger AUC and concordance are better. For ranking, ties receive half credit, and only folds with discordant outcomes contribute. For within-width concordance we pooled discordant pairs at the same retained width within each fold. Every metric averages eligible folds within task and then weights tasks equally.

| Source | Metric | Tasks / folds | Size | Size + vote | Paired difference [95% interval] |
|---|---|---:|---:|---:|---:|
| Path | Log loss | 14 / 42 | 0.63162 | 0.63367 | +0.00205 [−0.03005, +0.04885] |
| Path | Brier | 14 / 42 | 0.21750 | 0.21703 | −0.00047 [−0.01395, +0.01821] |
| Path | Ranking AUC | 12 / 28 | 0.78422 | 0.80843 | +0.02422 [+0.00604, +0.04542] |
| Path | Within-width concordance | 11 / 22 | 0.50000 | 0.58194 | +0.08194 [+0.02399, +0.14394] |
| Drawn | Log loss | 14 / 42 | 0.58047 | 0.56731 | −0.01316 [−0.03424, +0.00609] |
| Drawn | Brier | 14 / 42 | 0.19549 | 0.18929 | −0.00620 [−0.01576, +0.00256] |
| Drawn | Ranking AUC | 11 / 28 | 0.77913 | 0.82543 | +0.04630 [+0.00226, +0.10185] |
| Drawn | Within-width concordance | 11 / 27 | 0.50000 | 0.57327 | +0.07327 [−0.00054, +0.14968] |

Path folds contributed 617 discordant pairs in total, including 97 at the same width. Drawn folds contributed 697 and 137, respectively. The fitted coefficient on dissent was positive for every held-out task in both sources. Thus the combined model inherited the raw vote's ordering within each width. Across all widths, adding votes improved task-average ranking in seven path tasks, worsened it in one and tied in four. For drawn proposals, the counts were six, two and three. A better ordering can help prioritize a list. These outcomes alone do not measure how many refits a policy saves or which support it deploys. [Fold denominators](../../results/context_review/incremental-prediction/folds.csv) · [Task effects](../../results/context_review/incremental-prediction/tasks.csv)

The paired percentile intervals resampled eligible held-out task summaries 50,000 times (seed 744210). They describe variation among this finite collection of convenience tasks conditional on the fitted predictions. The training sets overlap, and there is no multiplicity adjustment. In a sensitivity analysis we omitted each task entirely, refitted the models on 12 tasks and evaluated each remaining held-out task. The 14 reruns gave task-average AUC changes from +0.01581 to +0.02707 on paths and from +0.02593 to +0.05463 on drawn proposals. Corresponding log-loss changes ranged from −0.02361 to +0.01373 and from −0.01723 to −0.00727. [Paired intervals](../../results/context_review/incremental-prediction/summary.csv) · [Refitting sensitivity](../../results/context_review/incremental-prediction/sensitivity.csv)

## Reproduction

Run from the thesis checkout with the archived refit-target output present.

```powershell
.\thesis-python.ps1 -m experiments.context_review.incremental_prediction --freeze-only
.\thesis-python.ps1 -m experiments.context_review.incremental_prediction
.\thesis-python.ps1 -m experiments.context_review.incremental_prediction_audit
```

The first computation stopped during the task-omission sensitivity when the optimizer reported a line-search failure. No comparison table had been written. The revised runner checks the objective's gradient directly (maximum absolute component at most 1e−7). The statistical design retained the same hash. The separate verification program reconstructed the proposal features and outcomes from the archive, fitted all 56 primary models with scikit-learn's Newton solver and recomputed the eight paired intervals. The two implementations' failure probabilities agreed within 2.1 × 10⁻⁸. [Initial lock](../../results/context_review/incremental-prediction/initial-protocol.json) · [Verification](../../results/context_review/incremental-prediction/verification.json)
