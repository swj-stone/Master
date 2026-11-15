# Independent R10 audit of nomination, stopping and cost

I inspected the ordinal decomposition, the matched-stopping implementation, their locked prediction inputs, and the Lean statements. The numerical comparisons below were reconstructed without calling the manuscript's selection or stopping functions, except when exercising those functions as targets of adversarial tests.

## Decomposing the nomination decision

The intermediate policy first maximizes the integer minimum vote, then maximizes the full margin among the highest-vote candidates. All three policies resolve any remaining tie in the archived candidate order. This intermediate policy can change a nominee only within the original winning vote tier. Unrestricted full margin can differ from this intermediate policy only by selecting a strictly lower vote tier. This is a path-specific descriptive decomposition chosen after observing previous results. It does not establish an optimal future tie policy.

I independently recovered the three policies from the stored four-panel tensors, sorted context margins, obtained guard and outer metrics from the rowwise forest probabilities, and applied fallback against each fold's own anchor. The equalities below hold record by record, rather than arising from cancellation across sources.

| Forest slate | Guarded records | Utility change within highest vote, points | Breach change within highest vote, points | Additional utility / breach change across vote tiers |
|---|---:|---:|---:|---:|
| Original, ten tasks | 30 | +3.444 | +6.667 | 0 / 0 |
| Original, six wider tasks | 16 | +5.491 | 0 | 0 / 0 |
| ANOVA, ten tasks, three draws | 90 | −2.222 | +2.222 | 0 / 0 |
| ANOVA, six wider tasks, three draws | 48 | +1.314 | +1.852 | 0 / 0 |

Allowing a strict vote reversal changed two nominees across the 184 comparisons. It changed the delivered support on original Parkinson fold 0 and decreased delivered AUC by 0.040816, leaving the utility and breach indicators equal. Both nominees in the other case, Steel fold 0 local draw 3, fell back to the anchor. A claim that strict reversals never affected the selected model or its predicted performance would therefore be incorrect.

## Exactness, edge cases and stopping

For a candidate, the smallest observed vote and smallest observed margin are separate upper bounds on the two final coordinates. They can be reached on different restorations. Taking the minimum of the lexicographic tuples directly over restorations would in general compute a different score. The implementation instead takes the minimum coordinate by coordinate and compares the resulting pair in lexicographic order. A losing or tied upper bound can eliminate the current candidate because an earlier incumbent has fixed tie priority. A trial still capable of beating the incumbent must continue. The rule applies to deterministic first-tie selection; recovering the entire tie group would require continuing at equality.

The global early exit is valid only after ordinal-only selection has a completed minimum vote of four, the universal maximum over four panels. An individual candidate's zero is an exact ordinal-only score, but cannot terminate the global search. A candidate with four votes can still improve its margin relative to another candidate with four, and a candidate with zero votes can still improve its margin against an incumbent with zero. The code correctly reserves both ordinal shortcuts for ordinal-only selection.

I compared the implementation against independent exhaust-all minima on **238,214** small scans, including repeated vote and margin values, negative and positive margins, differing visit orders with priority reset accordingly, and a running four followed by zero on the next context. All nominees agreed. The tests included a dynamic case in which a later four-vote margin legitimately wins and one in which a later zero-vote margin legitimately wins. Pure-margin equality with an earlier completed incumbent pruned correctly.

Lean 4.35.0 typechecked `formal/Challenger.lean`. Its statements prove nonincrease of natural-number minima and preservation of a losing lexicographic key under further coordinatewise minima. Only standard axioms were printed. Finite measured margins can be represented by order ranks preserving ties. The margin-to-rank representation, floating-point calculator, masked-model execution, global four-vote bound, wall clocks and the implementation itself are outside the formal theorem. The executable checks supply the link to these actual data.

## Mask accounting

I independently reconstructed the ordered empty, constant-weight separating, and full visible feature sets from feature ranks, deduplicated anchor and candidate masks in one set per policy/slate, and independently recalculated scalar and lexicographic scores using the original numerical tolerance. I used the separately calculated one-panel point scores for the local pipeline; scoring the first panel in a batched four-panel matrix can round differently near a tie. I checked all **1,288** stored method–slate winners and unique mask counts. Every record agrees.

| Original forest panel | Exhaustive family | Pruned ordinal | Pruned full margin | Pruned vote then margin | Point |
|---|---:|---:|---:|---:|---:|
| Ten tasks | 760 | 189 | 424 | 419 | 126 |
| Six wider tasks | 1,334 | 366 | 583 | 599 | 125 |

On the 30-fold original panel, the full margin needs five more distinct masks than the refined vote under matching stopping. On the 16-fold wide panel it needs sixteen fewer. Ordering or complexity comparisons between those two implementations are data dependent. A mask count is a number of distinct model predictions on a batch of proposal inputs, jointly cached across anchors and candidates within a run. It does not measure seconds, individual rows, or the entire feature-selection pipeline. In the local designs the seven policy runs start with fresh caches on each of the three slates, even when two draws use identical candidate sets. The data contain no slate with exactly one deletion alternative; eleven original slates have no alternative.

The timed path uses a new ballot and score cache for every method/repetition, deterministically shuffles the order of the seven methods within each repetition, and reuses one forest already fitted to that fold. The clock encloses ballot construction and actual masked predictions. It excludes fitting the shared model, construction of the anchor and slate, and the guard and outer refits. All methods even incur ballot construction on an original fold without a deletion. The protocol summarizes by summing per-fold/slate medians of three repetitions. A timing-summary regenerated from archived clocks represents arithmetic replay; fresh elapsed measurements require executing the forest voter with the original inputs.

## Completed timing replay

I compared all 3,864 positive elapsed-time records in `timings.csv` with the raw receipts for the 46 folds, checked the three repetitions and seven distinct method/order positions per slate per repetition, checked the timing and source digests, and reproduced every sum of within-slate medians in the published summary. Every live winner and cache size agreed with the independent offline recount. On the original ten-task and wide panels respectively, exhaustive margin took 2.7831 and 5.3226 seconds, exact-pruned margin 1.5444 and 2.2981, exact-pruned vote then margin 1.5490 and 2.3478, exact-pruned vote 0.6577 and 1.4029, and point 0.4539 and 0.4702. These all refer to nomination on the same machine, using a one-thread forest and unpinned CPU affinity. Differences of a few milliseconds between the refined and unrestricted margin have no established cross-machine significance.

## Frozen R10 disposition

I verified SHA-256 `B74947597D297FFC3E5A0552B0336F89B72044D81D8D6E00A107E4E49EE4A1E6` for `thesis-paper/build-robustness-r10/main.pdf`. **Weak accept for an empirical ML evaluation track.** The abstract limits its equivalence claim to guarded compression and breach on the tested slates. Lines 315–324 and Table 6 identify both across-tier nominee changes and the Parkinson AUC difference. Table 7 matches the independent mask recount. Table 9 and lines 642–650 distinguish the exhaustive margin from the three stopped methods and describe the measured interval, repetitions, aggregation and exclusions. The wording of the stopping lemma agrees with the formal result and fixed visitation order. No material scientific or cost-reporting issue remains in the reviewed scope.
