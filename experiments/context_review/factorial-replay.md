# What changed the fourteen failed deletion tests?

This retrospective replay fixes the fitted predictor, proposed deletion and reference support. Each proposal is checked on both ranking and guard samples, with the same masks available for both. For the endpoint family it tests the empty and full restorations. For the interior family it tests all sampled, previously unqueried restorations. A family fails if any restoration receives at most two affirmative panel votes out of four.

## Results and denominators

| Population | Count | Ranking endpoints fail | Ranking interiors fail | Guard endpoints fail | Guard interiors fail |
|---|---:|---:|---:|---:|---:|
| Every proposal with any novel probe | 216 | 190 | 207 | 185 | 200 |
| Ranking endpoints pass | 26 | 0 | 17 | 10 | 14 |
| Both samples' endpoints pass | 16 | 0 | 7 | 0 | 5 |

Among the 26 proposals passing the ranking endpoints, nine failed both the unchanged endpoints and the interior family on guard. One failed the guard endpoints alone, five failed the guard interiors alone, and eleven failed neither on guard. The previous count of fourteen thus conflated two comparisons. Only five of the fourteen had both guard endpoints pass. Among all 26, the full novel family has 12–16 masks depending on the split. Giving it just two distinct draws yields a mean failure probability of 32.44% on ranking and 35.48% on guard. These exact means average all equally likely pairs of the stored interiors separately within each proposal. The two guard endpoints fail on 38.46%. The area from which a mask is sampled and its cardinality still differ between the two families.

| Task | Distinct trajectory proposals across seeds | Sampled | Eligible | Ranking endpoints pass | Of these, guard endpoints fail | Of these, guard interiors fail | Both endpoint sets pass | Both pass and guard interiors fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Abalone | 60 | 58 | 0 | 0 | — | — | — | — |
| Credit Card Fraud | 38 | 38 | 38 | 10 | 2 | 4 | 8 | 2 |
| Gisette | 180 | 60 | 60 | 9 | 4 | 4 | 5 | 1 |
| ISOLET | 940 | 60 | 60 | 4 | 3 | 4 | 1 | 1 |
| Madelon | 26 | 26 | 26 | 0 | — | — | — | — |
| Spambase | 32 | 32 | 32 | 3 | 1 | 2 | 2 | 1 |

The archived sample comprised up to twelve evenly spaced proposals among distinct nonempty supports in order of first encounter on each nested search trajectory. Sampling fractions ranged from 60/940 on ISOLET to 26/26 on Madelon. Selection depended on previously observed ranking votes that determined each trajectory. All 58 Abalone proposals lacked eligible interior probes. The five remaining guard-interior failures after both sets of endpoints passed came from four tasks. Nested, separating and size-matched random ranking checks flagged one, four and four, respectively. Separating and random flagged the exact same four. Among the other eleven, they flagged one, two and two, respectively. These are counts against a finite, sampled definition of failure. They do not estimate harm after model refitting.

Across 5,124 equal-cardinality pairs of novel masks, 1,039 pairs had different decisions on ranking. Of these, 236 differed in the same direction on guard, 81 reversed and 722 tied. A tie here means the two masks produced the same binary decision on that sample; each individual 2–2 panel vote counts as a veto. On guard 115 out of 216 proposals had at least one discordant pair. This is an ever-observed event across differing numbers of dependent checks. The sampling units share rows, features, masks, seeds and earlier search history. Counts and matched probabilities are descriptive. No independent-proposal confidence intervals or hypothesis tests are supplied.

## Replay

Run from the master's checkout using the existing environment:

```powershell
.\thesis-python.ps1 -m experiments.context_review.factorial
.\thesis-python.ps1 -m experiments.context_review.factorial_report
.\thesis-python.ps1 -m experiments.context_review.factorial_weights
```

The first script reads the archived predictions, labels, source model traces, references and panel weights. It independently regenerates the sampled proposal identities and masked-support lookup, verifies that ranking and guard row IDs are disjoint, checks sampled labels against the hash-pinned validation data and excludes every design mask from the novel set. It computes weighted AUC from the empirical distribution of negative-class scores, averaging the two CDF limits at ties. It uses scikit-learn average precision on rare binary panels and weighted MAE/clipped R² on the regression task. All 1,918 archived rule-by-proposal decisions match, including ranking and guard status, ranking and guard interior failure counts and the original available nested panel ballots. The primary [replay design](../../configs/context_review/factorial.json) was fixed for this retrospective analysis after the original 14/26 results had been inspected. The original reserved-row split is a replay on already studied datasets.

The output directory is `results/context_review/factorial`. Files include full candidate-level indicators, the four-cell joint distribution, paired switches, exact matched-budget probabilities, same-size pair agreements and source hashes. Each method evaluates the same samples and masks within a split. Changing the panel partition also changes the associated four weight vectors. In the primary cross-tab fixed-panel comparisons among masks identify decision dependence on mask identity/cardinality. Switching ranking to guard while fixing masks identifies sensitivity to the combined sample and weights. The supplementary weight replay evaluates changes to weighting and resampling separately.

## What can be attributed to bootstrap weights?

The [weight sensitivity protocol](../../configs/context_review/factorial_weights.json) was written after the primary crossed counts above were known. All 26 original proposals remain in the denominator regardless of what the new weights decide. On the ranking sample these proposals all passed the original endpoints by construction. Keeping the row IDs, fitted predictions, masks, original point panel and strict 3-of-4 threshold fixed, redrawing only the three stratified bootstrap weight vectors caused 4, 5, 6, 10 and 7 of the 26 endpoint tests to fail under five prespecified draws. On guard the corresponding numbers were 11, 11, 10, 12 and 10, compared with 10 for the original weights. The new draws measure sensitivity conditional on the original screen.

Using only the unresampled point panel and holding class masses fixed at their two-sample mean, the ranking and guard endpoints failed on 2 and 8 of the 26 proposals and the corresponding interiors on 13 and 14. This comparison holds the metric functional fixed while changing cases. It uses a one-vote rule. It does not identify a decomposition of the production four-vote decisions into additive row effects and weight effects. The five bootstrap repetitions reuse the same rows and supports.
