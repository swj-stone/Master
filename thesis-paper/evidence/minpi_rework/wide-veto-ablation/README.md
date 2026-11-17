# Does the minimum add value to wide-source ranking?

The four wide-source outcomes had been opened before this mechanism test. We fixed the contrasts at `c89a106` and sealed all 20 selections at `21f682f`. The experiment replayed the same fitted forest, voting and acceptance partitions and input universe. It added three ranking paths to the two originally studied. Pooled greedily minimizes the unweighted mean of the p availability-size losses, intact greedily minimizes the loss at size p, and static orders features once by their individual intact deletion losses. All five paths faced both original acceptance guards. For each partner the structural portfolio chooses between its path and MDI, first by reachable width, then by worst acceptance loss and finally in favor of MDI. The accompanying files also include the original nominal-width tie rule and standalone paths. [Protocol](../../../../experiments/minpi_rework/wide_veto_ablation.md) · [selection hashes](lock.json)

## Physically referenced inputs

| Source | MDI | MDI + MinPI | MDI + pooled | MDI + intact | MDI + static |
|---|---:|---:|---:|---:|---:|
| Internet Advertisements | 149.6 | 127.6 | 129.6 | 127.6 | 132.2 |
| QSAR oral toxicity | 256.0 | 213.4 | 189.8 | 192.8 | 212.4 |
| Tehran Residential Building | 30.4 | 29.0 | 30.4 | 30.4 | 30.4 |
| UJIIndoorLoc | 18.6 | 18.2 | 18.2 | 18.2 | 18.6 |

The minimum portfolio was narrower than the pooled portfolio in seven jobs, wider in five, and tied in eight. Its source-weighted geometric width ratio was 1.009 (four-source bootstrap interval .967–1.070). The corresponding ratios against intact, static and MDI were 1.007 (.967–1.057), .965 (.959–.971) and .893 (.826–.965). In absolute columns the minimum portfolio averaged 5.05 more than pooled (−1.70 to +17.35). The largest disparity was on toxicity, where average widths were 213.4 with the minimum and 189.8 with the mean. Each interval resamples whole source means after averaging paired differences or log ratios within source.

Relative to pooled, the minimum portfolio's held mean AUC changed by +.00075, accuracy by −.00009, R² by +.00103 and scaled MAE by −.00061. Each predictive metric averages only two source means. Its change in held worst-size excess divided by the held null loss was +.00018 (−.00053 to +.00090). Both portfolios exceeded the .02 outer quota in 4/20 jobs, compared with 3/20 for MDI, 5/20 for intact and 5/20 for static. The four-source compression gain against MDI can be replicated with a pooled path. The comparison does not establish a width or predictive advantage specific to the minimizing vote.

## Held worst-size excess at a common feature budget

For each source and seed we also gave every ranking the same nominal number of columns, using the count selected by the independent MDI-only acceptance guard. Each table entry is the five-seed mean of the maximum over all acquisition sizes of the held proper-loss increase divided by the held null loss, including the p/s availability normalization. Lower values are better. The fixed budget can give unequal physical widths. For example, on toxicity MinPI used 226.2 reachable inputs against 269.0 for pooled at an identical mean nominal budget of 269.0.

| Source | MDI | MinPI | Pooled | Intact | Static |
|---|---:|---:|---:|---:|---:|
| Internet Advertisements | .0191 | .0096 | .0096 | .0095 | .0098 |
| QSAR oral toxicity | .0157 | .0113 | .0079 | .0090 | .0114 |
| Tehran Residential Building | .0122 | .0216 | .0561 | .0617 | .0807 |
| UJIIndoorLoc | .0164 | .0420 | .0416 | .0455 | .0539 |

At fixed nominal width, MinPI minus pooled averaged −.00766 (−.02574 to +.00257). The largest improvement, −.03448, came from Tehran; the five fits there reuse a single outer quarter block sequence. Its allocation crossed the .02 quota in two jobs versus five for pooled. On toxicity the minimum increased the mean worst excess by .00346 and on UJI by .00046. Against MDI the paired mean change was +.00526 (−.00697 to +.01809). The MDI budget was chosen to satisfy MDI's acceptance criteria, while the other rankings were forced to that budget. This comparison addresses predictive risk at a given nominal acquisition budget; it does not replace guarded portfolio comparisons. All baseline MDI and MinPI score fields replayed exactly over the 60 previously evaluated wide-source rows.

All 460 outcome rows include clean held predictive scores, worst-size loss, each support's actual and nominal width, and its winning path. To reproduce with the original receipts restored, run `python -m experiments.minpi_rework.wide_veto_ablation collect` followed by the same command with `score`. [Per job](scores.csv) · [source means](source-means.csv) · [paired effects](contrasts.csv) · [width ratios](width-ratios.csv)

## Independent referee assessment

The mathematical and computational question is defensible: conditional on an exchangeable size law and a fixed, first-missing-split forest, one can integrate the proper loss exactly and certify which availability sizes can be skipped during a greedy search. The union indicator representation of set functions and symmetrization over fixed-size slices have classical precedents. Attribute them explicitly and identify the forest-specific loss coefficients, update procedure and timing evidence as the contributions. [Set-function representations](https://doi.org/10.1287/moor.25.2.157.12225) · [Minsky–Papert symmetrization, Lemma 3.2](https://homepages.cwi.nl/~rdewolf/publ/qc/polynomials.pdf)

I would reject a general feature-selection superiority claim. The MDI comparison alone confounds adding a second candidate path with its voting rule. Pooled and intact portfolios match or exceed the average width performance across the studied panels. Total runtime against MDI, the held regression losses compared to the full model, four independent wide source identities, and a single measured outage network all limit the deployment case. The exact early-veto timing result, the controlled acceptance-rule effect on the seven-source panel, and the ability to compute the same forest model using fewer structurally referenced columns can be reported with their scopes. Evidence for the ranking rule itself remains mixed. Further independent evaluation with a justified acquisition law and a named acquisition/pruning alternative would especially matter for a claim about practical superiority.
