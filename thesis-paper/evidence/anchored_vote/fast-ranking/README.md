# Exact split-vote ordering on the four development anchors

The optimized implementation represents each positive binary64 threshold weight as an integer on a shared power-of-two denominator. Let the matrix entry at degree d and column j be the sum of the live integer weights at thresholds of degree d represented by j. For size k, multiply row d by the integer binomial coefficient C(n − d, k) and sum over degrees. Divide by the sum over columns to get relative pivotal credit. The common denominator C(n − 1, k) cancels. Clearing the at most three positive normalization denominators compares mean and minimum credits with exact integer arithmetic. Once a threshold gains a representative its incidences leave the matrix permanently.

The minimum admits a second saving. Price the largest tested size first, complete its leader across the other nonzero sizes, and retain rivals whose first share reaches the leader's completed minimum. The first share upper-bounds each rival's final minimum. Keeping equality preserves the specified Gain and index tie breakers. For one live multiplicity of one, the remaining private threshold weights can be sorted once.

The common sketch builder packs the Boolean threshold rows into bytes with the first sample as the high bit, deduplicates the byte keys in lexicographic order and returns the corresponding original rows. Classification and regression checks covered constant features, repeated columns, native missing values, close float32 thresholds, 512-row truncation and byte-boundary lengths. All 36 fixture sketches and four real sketches matched the frozen voter matrix and threshold weights element for element. Both ranking rules gave the same complete orders on the four real anchors. Direct subset enumeration agreed with the integer algorithm on 130 games under both rules, with and without the early exit. The prior floating implementation disagreed on three of 260 rule/game cases in the two extreme-scale fixtures. Those discrepancies appear in `oracle.json`.

Times below are medians of 21 rotated measurements on Windows logical processor 12 with one computational thread. These four sources were already open for development. Each ranking measurement starts before creating the 512-row sketch on the cached fitting inputs, given the same fitted model's gain values and previously measured anchor. Ranking finishes when the full order is available.

| Source | Anchor columns | Frozen sketch and minimum order (ms) | Packed sketch and exact minimum order (ms) | Candidate-size evaluations, before → after minimum bound |
|---|---:|---:|---:|---:|
| Bankruptcy | 76 | 92.14 | 18.09 | 7,560 → 3,824 |
| Forecast | 14 | 15.08 | 1.98 | 231 → 99 |
| Posture | 26 | 31.61 | 4.49 | 1,023 → 545 |
| Turbine | 2 | 2.50 | 0.54 | 4 → 3 |

At fixed sketch construction, the branch bound changed integer ordering time from 7.89 to 6.75 ms, 0.48 to 0.51 ms, 1.35 to 1.58 ms, and 0.15 to 0.15 ms, respectively. Most of the measured ranking gain came from faster sketch construction. Both the mean vote and minimum vote can use that implementation. The timing protocol leaves model fits and validation proposals unchanged.

Six further rotated repeats measured each selector from its first native model fit through final prediction. Every support, model-fit count, floor, selected utility and prediction matched its archived CPU counterpart. Each row's packed minimum uses the same model fits as the frozen minimum.

| Source | Frozen MinPI (s) | Packed MinPI (s) | Matched anchor Gain (s) | Retained, MinPI / Gain | Paired median time, packed minus frozen (s) |
|---|---:|---:|---:|---:|---:|
| Bankruptcy | 1.247 | 1.228 | 1.069 | 57 / 38 | −0.044 |
| Forecast | 0.430 | 0.411 | 0.422 | 14 / 14 | −0.018 |
| Posture | 2.927 | 2.776 | 2.221 | 26 / 26 | −0.077 |
| Turbine | 0.088 | 0.087 | 0.086 | 2 / 2 | −0.001 |

The paired median time improved over the frozen minimum in all four cases. Only forecast simultaneously kept as few columns and ran faster than the matched short Gain search. Repetitions measure runtime with identical training inputs at seed 42. They provide no additional predictive validation.

To reproduce, run the following commands inside the thesis checkout.

```powershell
.\thesis-python.ps1 -m experiments.anchored_vote.check_fast --repeats 21 --cpu 12
.\thesis-python.ps1 -m experiments.anchored_vote.check_fast --pipeline-only --repeats 6 --cpu 12
```
