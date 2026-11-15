# Choosing within the maximum-vote group

Each candidate in a given fold and draw has a worst-restoration vote count. The experiment conditions on the candidates attaining the largest such count. Four policies select, in turn, the earliest eligible candidate in the frozen order, one eligible candidate uniformly, the eligible candidate with the largest standalone point margin, and the eligible candidate with the largest full margin. All deterministic ties follow the frozen slate order. Uniform probabilities apply before guard evaluation, so several losing nominees can all contribute probability to the same anchor fallback.

The random forest supplies all masking, guard and final scores. The original design uses the same Gain-based slates as the crossed-learner experiment. The local design uses the ANOVA-based slates for each of the three recorded neighborhood draws. The guards, anchors, outcomes and candidate sets were measured before the reviewer requested this comparison. No threshold or winner was chosen using the new comparison. The policies have one guarded candidate nomination. The reported uniform expectation can be computed because all the candidate refits were archived.

| Slate | Panel | Tied top group | At least two guarded returns | Variation in guarded compression | Variation in breach |
|---|---|---:|---:|---:|---:|
| Gain | Ten | 14/30 | 9/30 | 5/30 | 6/30 |
| Gain | Wide | 10/16 | 7/16 | 5/16 | 5/16 |
| Local | Ten | 60/90 | 43/90 | 17/90 | 4/90 |
| Local | Wide | 42/48 | 37/48 | 25/48 | 13/48 |

The [policy table](../../thesis-paper/evidence/within_tier/means.csv) gives compression within tolerance, breach, AUC and accuracy. The [paired differences](../../thesis-paper/evidence/within_tier/contrasts.csv) integrate over the tie lottery and then resample whole sources. They average the three local draws within fold before aggregating folds within source. The source counts in the uncertainty calculation are ten and six. Each [per-slate bound](../../thesis-paper/evidence/within_tier/fold-bounds.csv) adds the positive and negative masses of the return-probability difference, weighted by the removed feature fraction. This gives a conservative interval for the possible difference in effective compression before observing the outer scores. [Source-level policies](../../thesis-paper/evidence/within_tier/tasks.csv) · [Observed variation](../../thesis-paper/evidence/within_tier/opportunities.csv)

In the original ten tasks, random choice recovered 2.53 points of the 3.44-point full-over-original gain. Full and point both added 0.91 points over uniform. Original full exceeded uniform within Dry Bean by 18.75 points and fell below it within Steel by 9.62 points. Original wide full exceeded uniform on Ozone, Parkinson and Semeion and fell below it on Urban. On the three local draws averaged within fold, full minus uniform was −0.25 points on the ten tasks and +1.26 points on the wider tasks. All four task-bootstrap intervals included zero.

Run these commands from the thesis checkout to reconstruct the complete tables and check all 736 policy expectations using exact rational arithmetic for effective compression and breach.

```powershell
.\thesis-python.ps1 -m experiments.context_review.within_tier replay
.\thesis-python.ps1 -m experiments.context_review.within_tier_audit
```

The local predictor archives contain standalone single-panel point scores for every candidate. The Gain predictor archives contain the four-panel tensor and the original unconstrained point winner. Single-panel and batched metrics can differ in the last bit. The replay uses the separately recovered Gain scores and checks every original winner. Running `within_tier points` with the original feature matrices regenerates those values using the original forest seed, estimator and single-panel measurement. The portable replay requires only archived predictions and outcomes.

The [ties-only sensitivity](../../thesis-paper/evidence/within_tier/ties-only.csv) conditions on a visible group of at least two candidates before the guard. It omits folds without an eligible draw, averages eligible draws within each included fold and retains sources with at least one eligible fold. It is a conditional analysis prompted by the observed numbers of ties. On the original wide panel the full-minus-uniform gain reaches 9.50 points with a paired interval from −2.77 to 23.17. No panel's full-minus-uniform compression interval excludes zero under this conditioning either.
