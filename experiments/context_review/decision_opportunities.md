# How often could a different nomination improve the decision?

This retrospective analysis uses the complete slates and the original guard. It includes all 46 external folds, eleven of which have singleton anchors. The nomination scores come from the same frozen tree on both receiving learners. A receiving learner trains its own guard, candidate and anchor.

## Comparing rules that respect the same vote

All three tie rules select among candidates with maximum integer vote. The original rule chooses the first candidate in Gain order. The refined rule maximizes the one-panel point margin within the tie, resolving any subsequent tie by the original order. The uniform tie rule assigns the same probability to each maximum-vote candidate. On a singleton each returns the anchor. We average the actually guarded outcomes of the tied candidates to obtain the exact mean of the randomized rule on each fold.

| Receiver and panel | Uniform within tie, U (%) | Point within tie, U (%) | Point minus uniform, pp [95%] | Uniform within tie, breach (%) | Point within tie, breach (%) | Mixed-utility best groups / actionable folds |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost, ten tasks | 6.16 | 7.29 | +1.14 [−1.05, 3.39] | 8.47 | 6.67 | 8/20 |
| XGBoost, six wide tasks | 9.49 | 7.34 | −2.15 [−6.45, 1.85] | 13.89 | 16.67 | 5/15 |
| Logistic, ten tasks | 6.38 | 5.78 | −0.60 [−2.92, 1.11] | 3.61 | 3.33 | 6/20 |
| Logistic, six wide tasks | 14.30 | 10.83 | −3.48 [−12.50, 2.76] | 13.19 | 16.67 | 4/15 |

Each task receives the same weight after averaging its folds. The intervals give the 2.5th and 97.5th percentiles of 30,000 paired whole-task bootstrap draws. A mixed group contains at least one candidate that passes both the guard and outer criteria and at least one that earns zero utility. The five wide XGBoost folds were Parkinson 0 and 1, SECOM 2, and Semeion 0 and 2. Respectively 3/8, 2/8, 1/2, 2/8 and 2/8 maximum-vote candidates passed both checks.

## What could have changed after the guards?

For a fold, let `p_s` and `q_s` be the probabilities that two policies return support `s` after their common guard. Let `d_s` be the fraction of anchor columns removed and `b_s` the indicator that the refit meets the outer tolerances. The paired difference in expected utility is

```
sum_s c_s b_s,    c_s = (p_s - q_s) d_s.
```

Before opening the outer labels, this sum lies between the sum of its negative coefficients and the sum of its positive coefficients. Common support mass cancels. The anchor has zero saving. This argument also applies to deterministic policies, whose returned distributions each place mass one on a single support. The endpoints allow the pass indicators of different supports to vary freely. A shared set of outer labels and the fitted predictions may restrict which indicator combinations can occur.

| All-context minus point | Outcome-free mean U envelope (pp) | Tasks permitting any positive change | Probability a task bootstrap omits all such tasks |
|---|---:|---:|---:|
| XGBoost, ten tasks | [0, +3.33] | 2/10 | 0.1074 |
| XGBoost, six wide tasks | [−2.78, +2.78] | 1/6 | 0.3349 |
| Logistic, ten tasks | [0, +1.67] | 1/10 | 0.3487 |
| Logistic, six wide tasks | [0, +2.61] | 1/6 | 0.3349 |

When exactly `k` of `n` tasks permit a positive change, a bootstrap of `n` whole tasks omits all such tasks with probability `((n-k)/n)^n`. For the primary wider XGBoost comparison, different final supports were returned on SECOM fold 0 only. Both removed half the anchor and the fold had weight 1/18. This fixed the largest possible gain at 2.78 points before any outer scoring, below the prespecified five points. The observed zero difference therefore says little about ranking when the guarded policies return substantially different supports on many tasks.

## Reproduction

Run `./thesis-python.ps1 -m experiments.context_review.decision_opportunities` from the thesis checkout. With the published archive extracted, run the same module using the documented reproduction environment. It checks the sealed selection and receipt hashes, computes the envelopes using the original guard information, then scores the original candidate probability vectors against the separately sealed outer labels. It writes the six tables in [the evidence directory](../../thesis-paper/evidence/opportunities/). Field `vote` denotes the original integer rule with Gain resolving ties, `refined` the point-within-vote rule and `top_uniform` a uniform choice within the best vote group.
