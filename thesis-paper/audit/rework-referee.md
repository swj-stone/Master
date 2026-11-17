# Independent review of the compression pilots

**Decision, 17 November 2026.** I would reject a general performance claim for the stagewise GCV and triplet paths. A computational equivalence claim has a defensible mathematical target. Its end-to-end advantage needs separate evidence. I did not examine any fresh outer outcomes.

## Results I could reproduce

I paired rows by task, source and seed, checked uniqueness, averaged within each source and gave every source in a domain equal weight. Width and time ratios are geometric. Time ends after the final fit and test prediction. Predictive differences subtract the comparator from the method. These are development comparisons.

| Rule and receiver | Comparator | Domain, sources | Width ratio | End-to-end time ratio | AUC / accuracy change or R² / scaled MAE change |
|---|---|---|---:|---:|---:|
| Stagewise signed GCV, ridge | Single-coalition continuous GCV | Classification, 9 | 1.018 | 1.155 | −.0032 / −.0023 |
| Stagewise signed GCV, ridge | Single-coalition continuous GCV | Regression, 9 | 1.011 | 1.276 | −.0039 / +.0012 |
| Triplet minimum, neighbours | Pooled votes | Classification, 5 | .990 | 1.008 | +.0009 / +.0050 |
| Triplet minimum, neighbours | Pooled votes | Regression, 4 | 1.038 | 1.011 | +.0046 / −.0023 |
| Triplet minimum, ridge | Pooled votes | Classification, 8 | .917 | .990 | +.0000 / −.0061 |
| Triplet minimum, ridge | Pooled votes | Regression, 7 | 1.000 | 1.017 | −.0088 / +.0013 |

For stagewise GCV the descriptive 95% source-bootstrap intervals for time are [1.109, 1.203] in classification and [1.204, 1.348] in regression (50,000 resamples, seed 27512). None of the 18 source means met the registered time-or-width threshold and predictive margins against the single-coalition control. Full-width ridge comparison breached the predictive margins in six of nine sources in each domain. The zero-inner-loss, single-pass variant still lost .0126 AUC and .0142 accuracy on average against full ridge. Its regression width equaled that of its single-coalition control.

With a neighbour receiver, Raisin alone met the joint conditions against pooling. Its width ratio was .794, with accuracy down .0056. With a ridge receiver, cluster-resampled width relative to pooling had a descriptive interval [.817, 1.000] for classification. Landsat retained .630 times the pooled width with accuracy down .0052. Musk retained .794 times the width but lost .0436 accuracy. The remaining six classification sources had identical paired geometric widths. All seven regression sources had identical paired widths. For example, Forest Fires changed one support and mean R² fell .0597. Against the full ridge receiver, the domain averages lost .0118 AUC and .0274 accuracy in classification, and .0212 R² in regression. The neighbour and ridge pilots used different source lists. Receiver sensitivity needs the common-source comparison.

## What the pilot can establish

* The integer ballot implementation has two exact checks. A triplet cannot flip when its current margin lies outside the interval bounded by all available one-feature changes. A partially evaluated candidate has a worst-context score no greater than the smallest score observed so far. An independent direct evaluation of every row and feature matched the eager and lazy winner on 1,000 generated cases, including ties and zero ballots, with the implementation current at review. This supports testing deterministic equivalence across complete real paths.
* The observed use of a minimum across three particular nested prefixes provides evidence on those prefixes. A claim that the greedy rule solves the original minimum over all coalitions, or guarantees generalization of retained features, would need a different argument. The common-anchor quota identity makes comparisons to the identical continuous minimum algebraically redundant.
* A speed study can answer how rapidly the **same** problem is solved. The fair comparator is an efficient eager calculation with shared precomputation and the same tolerances and tie-break. Report full-order, chosen-mask and prediction equality, implementation/runtime, candidate–context evaluations, and ballots actually inspected. The current triplet work counter counts candidate–context calls. The number of rows evaluated per call varies; the counter cannot measure the number of triplet votes or prove a time reduction. The registered compression criterion against a competitive standard selector remains a separate test.

## Experimental and reporting issues

* At review, the triplet production import failed after a shared module was moved into the experiment directory. I used an import alias only for the mathematical cross-check. Validate from the installed project's normal entry point once repaired. The pilot settings digest covers the backward path even for a triplet run and omits the triplet ranker. Predictions, supports, the actual active ranker, neighbour construction, receiver, data and protocol must be bound into the final manifest.
* Class-matched anchor/hit/miss triplets use the nearest opposite-label neighbours pooled over other classes. Regression uses training-target quantile classes. Describe the continuous-margin comparator as the actual matched triplet estimator. A comparison against a standard multi-class ReliefF or RReliefF would require its documented class weighting or continuous-outcome treatment.
* The chosen feature path uses three sizes along the same path. The matched full-context and pooled controls have direct bearing on the argument for this choice. A comparison solely with recursive elimination confounds voting with implementation cost and ranking strategy.
* Fresh source/seed masking, grouping by participants or material where available, removal of classifier outputs, a single chronological holdout, and preservation of task-level dependence are good design choices. Four sources per domain support a descriptive panel contrast. Reusing the same chronological test at five seeds does not supply five independent temporal holdouts. No clinical accuracy, population noninferiority, or cross-domain generality claim follows. Source-level bootstraps preserve dependence among seeds but cannot undo purposive sampling, pilot selection, or repeated viewing of development outcomes.
* Before exposing held-out outcomes, retain a pre-registered target, matched controls, the direct-strong selector, finite complete run grid, prediction hashes, inclusions, plus fixed splits and source IDs. The scoring interface alone checks only the supplied file digest; an independently written digest at scoring time would not establish a pre-score lock. Report both selection and end-to-end timing as stipulated in the protocol. A favourable post-hoc source result can motivate a future scoped hypothesis.

The source computations above used `results/minpi_rework/path-dense-uci-five/scores.csv`, `path-zero-uci-five/scores.csv`, `triplet-path-knn/scores.csv` and `triplet-path-ridge/scores.csv`. The dense GCV studies contain five seeds per source and the triplet studies contain three.
