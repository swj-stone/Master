# Counting and timing exact early stops under the same traversal

The earlier forest timing enabled early termination for the ordinal rule and exhausted the continuous rule. Both minimize a scalar over restorations. We reran each rule with its own exact stop on the same ordered candidates and restorations. A third rule maximizes the ordinal minimum and uses the continuous minimum to break ties. These are two separate minima across the same contexts. Applying a single lexicographic minimum to pairs of context scores gives a different rule.

Suppose the seen contexts give running minima $v_t$ and $m_t$. Every completion has $v ≤ v_t$ and $m ≤ m_t$, hence its lexicographic key is at most $(v_t,m_t)$. If this bound is weakly below the incumbent, the candidate cannot win because ties favor the earlier candidate. The same proof applies to either scalar. An ordinal zero can end that candidate's evaluation, and an incumbent with four votes ends ordinal nomination. A four-vote incumbent can still lose a margin tie-break. The test suite independently exhausts small slates, including cases where the two worst contexts are different. [Proof](../../thesis-paper/formal/Challenger.lean) · [Exactness tests](../../tests/test_matched_stopping.py)

The traversal preserves the initial Gain or ANOVA candidate order, visits the empty restoration first, and uses the same order as the stored separating family thereafter. Within each nomination, the anchor and candidate share a cache indexed by the actual visible feature indices. Every distinct cached support costs one prediction on the proposal panel. A point nomination uses the empty restoration and the separately scored original validation panel. Every comparison starts with a new score cache.

| Number of distinct masks, summed across folds | All contexts | Early-stop vote | Early-stop margin | Early-stop vote then margin | Point |
|---|---:|---:|---:|---:|---:|
| Original Gain, ten tasks | 760 | 189 | 424 | 419 | 126 |
| Original Gain, six wider | 1,334 | 366 | 583 | 599 | 125 |
| ANOVA draw 1, ten | 1,041 | 450 | 680 | 686 | 168 |
| ANOVA draw 1, wider | 1,746 | 548 | 1,013 | 1,013 | 144 |
| ANOVA draw 2, ten | 1,041 | 450 | 716 | 720 | 168 |
| ANOVA draw 2, wider | 1,746 | 560 | 1,039 | 1,041 | 144 |
| ANOVA draw 3, ten | 1,041 | 469 | 705 | 714 | 168 |
| ANOVA draw 3, wider | 1,746 | 526 | 1,038 | 1,019 | 144 |

The 46 original folds and three local candidate draws on those folds give 184 slates. For every slate, bounded enumeration reproduced unrestricted nomination separately for all three keys. Early-stop voting used fewer masks than early-stop margins on 155 slates and the same number on 27. It used more on Parkinson fold 2 for Gain (70 versus 63) and ANOVA draw 1 (84 versus 63). Counts over these repeated slates remain dependent. On the first local draw and original Gain folds, the exhaustive margin, single-point and stopped vote counts also reproduce the earlier independent implementations in all 276 fold–policy comparisons. [Fold-level choices and counts](../../thesis-paper/evidence/matched_stopping/folds.csv) · [Summed counts](../../thesis-paper/evidence/matched_stopping/summary.csv)

The live experiment fits one 100-tree single-worker forest per fold and times each nomination against this fitted model. Seven configurations each get three repetitions per slate, with a deterministic pseudorandom arm order within each repetition. The timer includes constructing the ballot, mask allocation, predictions, panel metrics and traversal. Feature ranking, input preparation, the common forest fit, guard and receiver fits are outside the timer. The summaries sum within-slate median times across folds. These measurements characterize the search stage on this machine. They cannot determine end-to-end speedups when training the models dominates. The host ran Windows 11 x64 with Python 3.13.9, scikit-learn 1.9.0 and NumPy 2.4.6. [Raw nomination times](../../thesis-paper/evidence/matched_stopping/timings.csv) · [Timing summary](../../thesis-paper/evidence/matched_stopping/timing-summary.csv)

| Summed nomination seconds | Exhaustive margin | Early-stop vote | Early-stop margin | Early-stop vote then margin | Point |
|---|---:|---:|---:|---:|---:|
| Original Gain, ten tasks | 2.783 | 0.658 | 1.544 | 1.549 | 0.454 |
| Original Gain, six wider | 5.323 | 1.403 | 2.298 | 2.348 | 0.470 |
| ANOVA draw 1, ten | 3.911 | 1.665 | 2.551 | 2.573 | 0.618 |
| ANOVA draw 1, wider | 7.279 | 2.239 | 4.370 | 4.300 | 0.565 |
| ANOVA draw 2, ten | 3.923 | 1.683 | 2.711 | 2.735 | 0.611 |
| ANOVA draw 2, wider | 7.254 | 2.322 | 4.399 | 4.404 | 0.554 |
| ANOVA draw 3, ten | 3.942 | 1.744 | 2.663 | 2.697 | 0.612 |
| ANOVA draw 3, wider | 7.319 | 2.149 | 4.473 | 4.371 | 0.562 |

The portable replay reads the frozen masked metrics, the archived order of candidate features, and the recorded live timings. A timing rerun additionally needs the fold feature files. Individual fold receipts under `results/context_review/matched-stopping` allow interruption and resumption. A fresh timing run uses a fresh run directory. To verify the distributed sample, run the first two commands below. To generate a new live sample, replay the mask schedule first, run the last command and then use the audit's `--record` option in the new working copy. The [timing receipt](../../thesis-paper/evidence/matched_stopping/timing-receipt.json) identifies the design, source and forecast locks, and includes both the original and line-ending-normalized checksums of the archived clock measurements.

```powershell
.\thesis-python.ps1 -m experiments.context_review.matched_stopping replay
.\thesis-python.ps1 -m experiments.context_review.matched_stopping_audit
.\thesis-python.ps1 -m experiments.context_review.matched_stopping time
```
