# Does dissent add information after continuous endpoint damage?

This retrospective diagnostic uses all 704 distinct archived XGBoost fold–support pairs from 42 folds of 14 PMLB tasks. Of these, 302 crossed the original refit-loss tolerance. The adaptive search path supplied 400 offers (189 failures) and uniform deletions supplied 379 (163 failures). The collections share 75 identical supports and 50 failures. One observation links a proposed reduction to a freshly refitted anchor on a common outer test fold. Features come from the fitting and ranking partitions. Previously inspected outer outcomes serve as the target.

The design in `configs/context_review/matched_signal.json` was committed at `8748d0f` before these fits. We hold out one whole task and fit on the other 13. Seven ridge-logistic specifications share the same weighting, penalty, feature scaling and held-out observations. We give tasks equal weight, then give folds equal weight within each task and offers equal weight within each fold. No parameter was tuned. The starting model uses four width-spline columns. The next model adds the fraction of the anchor's **total split gain** deleted and the fraction of its linear rank weights deleted. The rank comes from descending **average split gain** in the full training-only model. Both importance types were recovered by refitting the full model on the original training partition. All 42 reconstructed orders matched exactly.

For each panel, extension and metric, continuous damage equals (masked anchor score − masked candidate score − the metric's relative allowance × absolute anchor score) divided by the metric's nominal allowance (.03 for AUC/accuracy and .05 for applicable rare-class AP). The maximum over metrics gives a signed, continuous measure for that panel and extension. The endpoint model adds the unresampled empty-extension value and the mean and maximum across both endpoint extensions and the original four panels. We then add the largest fraction of dissenting panels over the separating contexts. The denominator consists of four repeated weights on the same ranking rows.

## Held-out task results

The estimates below compare the endpoint model with the same model plus dissent. Lower log loss and higher within-fold failure–pass concordance are favorable. Intervals resample the eligible task-level paired differences 50,000 times, conditional on the fitted predictions. All 14 tasks and 42 folds contribute to each log-loss result.

| Training → evaluation source | Offers | Endpoint log loss | With dissent | Paired change (95% interval) | Within-fold AUC change (95% interval) | AUC tasks / folds |
|---|---:|---:|---:|---:|---:|---:|
| Unique union → unique union | 704 | .5471 | .5391 | −.0079 [−.0216, .0073] | −.0165 [−.0469, .0030] | 12 / 30 |
| Path → path | 400 | .5742 | .5798 | +.0056 [−.0120, .0321] | −.0036 [−.0163, .0108] | 12 / 28 |
| Uniform → uniform | 379 | .5407 | .5228 | −.0179 [−.0373, .0020] | +.0224 [−.0118, .0735] | 11 / 28 |
| Path → uniform | 379 | .5758 | .5585 | −.0173 [−.0284, −.0056] | +.0009 [−.0091, .0112] | 11 / 28 |
| Uniform → path | 400 | .5808 | .5793 | −.0016 [−.0199, .0191] | −.0074 [−.0314, .0172] | 12 / 28 |

The union's within-fold AUC falls from .842 to .825 over 2,499 failure–pass pairs. Among its 322 same-width pairs in 28 folds of 11 tasks, concordance changes by −.0233 [−.0874, .0374]. Width and the two Gain variables already improve union loss over width alone by .0276 [.0088, .0436] and increase within-fold AUC by .0450 [.0079, .0940]. The width-only score ties every pair of equal width. Adding the two Gain variables increases union same-width concordance from .500 to .629. The endpoint model is the fixed, feature-rich comparator. Simpler models can score better for individual metrics and sources.

Adding all-context continuous mean and maximum damage to the endpoint model changes union loss by −.0018 [−.0089, .0077]. Adding dissent to that all-magnitude model changes loss by −.0086 [−.0239, .0088] on the union, +.0062 [−.0132, .0356] on the path and −.0215 [−.0420, −.0005] on uniform deletions. The latter is a secondary contrast drawn from overlapping proposal sets. These results support a source-dependent predictive association. They supply no direct evaluation of a decision to skip a refit.

## Common support and computation

In each evaluation fold, an offer enters the geometry comparison if its retained width occurs in the training source's collection for that fold and its deleted Gain mass lies within the observed range for that width. Held-task outcomes play no part in this rule. Feature identities, unmeasured proposal characteristics and distributions within the range can still differ. This use of both offer collections would require access to both during evaluation.

| Training → evaluation source | Inside range | Change in log loss (95% interval) | Exclusive offers inside range | Change on exclusive offers (95% interval) |
|---|---:|---:|---:|---:|
| Path → uniform | 116 / 379, 11 tasks | −.0232 [−.0422, −.0051] | 41 / 304, 7 tasks | −.0137 [−.0399, .0138] |
| Uniform → path | 128 / 400, 12 tasks | −.0247 [−.0539, .0044] | 53 / 325, 8 tasks | −.0194 [−.0680, .0249] |

On the 75 identical supports in 27 folds of 11 tasks, the augmented model trained on uniform deletions has log loss .369 against .442 for the path-trained model. The paired path-minus-uniform difference is +.0729 [.0122, .1402]. Only five tasks contribute failure–pass pairs to that ranking comparison, and four have identical within-fold AUC averaged over their eligible folds.

With reuse of every distinct mask within each fold and proposal collection, unresampled empty damage needs 746 mask evaluations on the union, the four-panel endpoint summaries need 1,492, and separating-context damage or dissent needs 5,539. The latter two signal types use identical masked predictions and can share their scoring work. Moving from endpoints to separating contexts adds 4,047 distinct evaluations over 42 folds, with zero extra evaluations in one fold. The corresponding endpoint/separating counts are 884/3,183 for the path and 842/3,079 for uniform deletions. These counts describe distinct evaluations of the already-fitted full model on ranking rows. Time, memory, refit savings and the cost of acquiring historical failure labels require separate measurements.

Sun and Raskutti estimate variable importance by starting from a dropout solution and allowing a limited number of learning updates. Their theory covers specific architectures, including boosting with symmetric weak trees. Our comparison uses fixed XGBoost masking scores and a cross-task risk fit to classify tolerance failures on candidate blocks. We did not reproduce their update algorithm or estimate an efficiency advantage over it. [Sun and Raskutti, AISTATS 2025](https://proceedings.mlr.press/v258/sun25a.html)

## Reproduction

From the thesis checkout with the archived primary, refit-target and rule-sensitivity runs present, execute:

```powershell
.\thesis-python.ps1 -m experiments.context_review.matched_signal
.\thesis-python.ps1 -m experiments.context_review.matched_signal_audit
```

The first command checks 273 input hashes, reconstructs the full model's split statistics, recovers continuous margins and verifies every dissent score against the original ballots. It saves all logits, fitted parameters and fold-level outcomes under `results/context_review/matched-signal-r2`. The initial run under `matched-signal` exposed a floating-point matrix-product difference for rows with identical width. The corrected run sums slopes in the same order for every row and requires exact .500 width-only concordance in every eligible fold. The design, features and targets were unchanged. The independent check refits all 294 models with a separate solver, obtaining a maximum probability difference below 3.4 × 10⁻⁷. It checks the 36 primary-contrast metric/coverage summaries, including paired interval endpoints.

The tracked `thesis-paper/evidence/matched_signal` directory contains the 704 feature and target records, per-task scores, every prespecified comparison, per-fold mask counts and verification receipts. Reported intervals describe heterogeneity across these available tasks. They hold the fitted models fixed and do not incorporate model-training uncertainty, multiplicity adjustment, or evidence from newly recruited tasks.
