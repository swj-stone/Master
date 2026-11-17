# Independent numerical review of the 13-source tree paper

The manuscript reviewed is commit `d338e31`. A separate timing revision is `90339b0`. This report does not use the outcomes of the additional five-source challenge or the subsequent repair experiments.

## Verdict

**Reject in the current form as a top-conference algorithm paper.** The empirical compression difference from the registered Gain sweep is real. The decisive control is a compiled partial-distance greedy solver that returns the very same supports, scores and deletion sequences. It is faster than the reviewed implementation with size-conditioned voting. The competitive single-ablation ranking costs much less than the original voting implementation and uses only 0.34 additional columns per source on average. Those findings support a technically credible evaluation of frozen-tree input compression but do not demonstrate that the three-size minimum is a competitive practical advance. A later revision should be judged against the same strong controls.

## Independently reproduced

I checked the 65 signed selection receipts against the lock. I verified the digests of their forecast archives and the 65 archives from the post-release ranking controls. I recomputed each metric from predictions and held-out targets for all 585 locked method/job arrays plus all 130 additional ranking arrays. All reported scores reproduced to numerical precision. The archived Madelon result also matches the manuscript: mean AUC 0.8302 for the supervised refitted Gain comparison and 0.7904 for the guarded greedy refit. SHA verification and prediction-level checks establish which experiments were evaluated. They do not change the exploratory status of the two methods added after the 13-source results were opened.

The weighted estimates below first average the paired effect across seeds within a source and then average the sources. The log scale is used for the width ratios. Absolute differences in columns are arithmetic means.

| Comparator | Greedy/comparator width | Mean difference in columns | Sources with smaller mean log width | AUC difference | Accuracy difference | R² difference | Scaled MAE difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Registered Gain sweep | .8939 | −2.985 | 10/13 | −.00247 | −.00146 | +.11973 | −.01531 |
| Full forest | .3984 | −30.262 | 13/13 | −.00372 | −.00385 | +.13619 | −.00009 |
| TreeSHAP ranking sweep | .9634 | −1.062 | 8/13 | +.00102 | +.00501 | −.09818 | +.01552 |
| Full-model ablation ranking sweep | .9877 | −.338 | 7/13 | −.00010 | −.00128 | +.00067 | −.00046 |

Ten sources used fewer inputs than Gain; the other three tied. The geometric reduction against the full-ablation sweep is 1.23%. The registered source-level margins versus Gain fail on Robot LP1. Versus the uncompressed teacher at least one margin fails on CT location, Dermatology, Mice, Robot LP1 and Superconductor. Against TreeSHAP the aggregate regression R² difference falls outside the .02 tolerance. Dropping Appliances changes the mean R² difference versus Gain from +.1197 to −.0031. Against TreeSHAP it changes −.0982 to +.0045. Both signs are consequently fragile. Music, Online News and Appliances have negative full-model mean R². Target-scale MAE must be reported per dataset.

There were 65 benchmark jobs and 57 distinct four-way index partitions. For Appliances and Online News the five repeats produced bit-identical arrays for all nine locked methods. Their five repeated jobs do not represent five independently varying predictive splits. Source-level uncertainty appropriately retains the repeats as one cluster, but the words “65 held-out splits” obscure the distinction.

## Timing and attribution

The 96,417 candidate visits include 3,255 initial norms. The three-size, full-size and no-direction methods required 1,041, 1,064 and 6,859 further norms, respectively. Limiting the calculation to the three-size rule saves 23 of 4,319 total exact norms compared with a single full-size vote, or 0.53%. The other saved recalculations are attributable to the full-size direction, and shared norm caching already avoids the majority of the candidate checks. The outputs of the exact partial-distance solver establish that the predictive and compression effects can be realized without any voting.

| Source-equal selector time ratio, three-size implementation / comparator | Manuscript revision d338e31 | Fused revision 90339b0 |
|---|---:|---:|
| Compiled partial-distance greedy | 19.77 | 3.57 |
| Compiled Gain sweep | 18.50 | 3.34 |
| Full-ablation ranking and sweep | 9.69 | 1.75 |
| Exact TreeSHAP ranking and sweep | 2.23 | .403 |

Against direct partial-distance search, the fused ratio is 3.57 with a source-bootstrap interval of 3.12–4.05. Its matched fit-inclusive ratio is 1.0065 (1.0035–1.0100). Provide the numerator, raw representative times, denominator and scopes; “the control was faster” conceals the scale. Do not compare the older four-bin compiled solver with the three-size game as if they were one implementation. The GPU/CPU routines have distinct setup costs. The principal cross-method times were assembled across runs on different logical cores, and they reuse measured shared fit and extraction durations. Near-unity pipeline estimates deserve an interleaved same-machine replay if interpreted as speed effects. The ratio of a search time to the nearly zero selector overhead of a full uncompressed teacher should be suppressed.

## Required corrections to the article

1. Describe 100 boosting **rounds**, maximum depth four. The four-class Robot LP1 model uses a tree per class per round. Give the response definitions alongside source names. Anuran predicts one frog family versus the rest, Mice predicts genotype, Dermatology targets class 1 versus the rest, Arrhythmia uses an abnormal-versus-normal endpoint, Music predicts latitude and Online News predicts log(1+shares). These are consequential adaptations of the original tasks.
2. Distinguish disqualification from winning a greedy step and failure of the fidelity budget. When a feasible incumbent lowers the cutoff below the budget, a projected lower bound above the cutoff can still belong to a feasible deletion. The current proof correctly uses the incumbent. Introduction and method prose should say so as well.
3. Put the fastest same-output implementation and full-ablation ranking in a numerical main-text comparison with scope-matched times, width and scores. Show the comparative limitations versus the original teacher and TreeSHAP, with source exceptions. Empirical source-bootstrap intervals are adequate as panel summaries; selected sources do not establish general predictive noninferiority.
4. A sufficient-condition proof for arbitrary projection directions establishes safety in exact arithmetic. The combinatorial anchor formula follows from a classical unanimity expansion. The two-feature example establishes strictness for small-background vetoes and cannot establish their practical importance. Retain the prior attribution and identify what raw feature savings remain after excluding inputs never used by the fitted teacher, if making acquisition claims versus all columns.
5. Replace the public-facing thesis README and archive references for the earlier manuscript. Supply an unambiguous path to the current paper, selection protocol, raw receipts or archive digests, measured controls, compiled solver and scoring commands. The five-page PDF reviewed resolves the in-text references and its table is readable. The main entry point available at the time of this review pointed to a different paper.

The five-source extension can provide an independent test of the post-release ranking controls and later variants if the hypotheses, eligible arms and predictions were fixed before its outcomes were opened. That is separate from this verdict on the original comparison.
