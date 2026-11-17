# Held-out results for adaptive spline refitting and single-input outages

Neither workflow met its registered primary success criterion on either held-out panel. All 65 source–seed runs produced forecasts. The panels contain eight original sources (four classification, four regression) and five wider sources (three classification, two regression). The predictions and executable protocol were committed before scoring. [Plan](../../../../experiments/minpi_rework/refit_confirm.md) · [Forecast seal](sealed.json)

Each source has equal weight after averaging its five paired seeds. Brackets give percentile intervals from 10,000 source-cluster bootstrap draws, conditional on these sources and splits. Small source counts limit their resolution, especially for the two wider regression sources. Repeated chronological tests use overlapping or identical dates. Widths and times are geometric mean paired ratios. Predictive metrics and losses are arithmetic paired differences. The main scores weight observations equally. [Split scores](heldout_scores.csv) · [All contrasts and intervals](heldout_contrasts.csv)

## Adaptive refitting

The screened selector searches a width-conditioned exchange game. Its primary comparator improves the same fidelity objective by single exchanges. Each arm refits a spline ridge receiver on its selected columns, searches the same width grid, and uses the same validation gate. Success requires the candidate to retain at least 10% fewer columns or take at least 10% less time with no increase in width, subject to both quality margins against both greedy and full. The time includes selection, refitting and generating test predictions. Each arm has three independently timed repetitions per split.

| Held-out scope | Retained width vs greedy | Total time vs greedy | ΔAUC / Δaccuracy or ΔR² / Δscaled MAE | Width vs full |
|---|---:|---:|---:|---:|
| Original classification | 0.923 [0.797, 1.128] | 1.240 [0.929, 1.919] | +0.0064 / +0.0047 | 0.374 [0.245, 0.573] |
| Wider classification | 1.505 [1.000, 1.861] | 2.676 [1.014, 4.628] | −0.0105 / −0.0164 | 0.174 [0.063, 0.451] |
| Original regression | 0.989 [0.968, 1.000] | 1.247 [1.002, 1.924] | −0.0154 / +0.0021 | 0.133 [0.024, 0.738] |
| Wider regression | 1.000 [1.000, 1.000] | 1.526 [1.255, 1.856] | +0.0036 / −0.0012 | 0.435 [0.379, 0.500] |

Only Coimbra met the joint criterion when applied separately to each source. Across seven classification sources, width and time ratios were 1.138 [0.895, 1.477] and 1.724 [1.086, 2.881]. Across six regression sources they were 0.993 [0.978, 1.000] and 1.334 [1.041, 1.781]. Both panel means for each task type stayed within the full-model predictive margins. The large original-panel mean R² gain of 0.480 vs full was concentrated in Appliances (+1.237) and Music (+0.689).

Against energy ranking, the adaptive selector used 21% fewer columns on the original classification panel (ratio 0.790 [0.687, 0.901]) and 15% fewer on the wider panel (0.847 [0.582, 1.046]), with mean AUC gains of 0.0120 and 0.0121. Its runtime ratios were 1.242 and 2.348. It met the width and quality thresholds against energy on wider regression only. Against GPU Gain ranking followed by the same spline receiver, the original classification and regression panels passed the joint criterion. Wider classification missed the AUC allowance (−0.0198), and wider regression retained more columns (ratio 1.329) and missed the R² allowance (−0.0219). The fixed-five sensitivity lost 0.048 and 0.075 in scaled MAE relative to full on original and wider regression, respectively (higher is worse).

As a descriptive application of the same margins to individual splits, the adaptive selector missed the full-model quality allowance on 13/35 classification tests (37%, source bootstrap 23–51%) and 2/30 regression tests (7%, 0–13%). Against greedy the counts were 5/35 (14%, 6–26%) and 1/30 (3%, 0–10%). The joint quality-and-efficiency criterion failed on 30/35 (86%, 74–94%) and 29/30 (97%, 90–100%). The validation gate therefore does not certify an outer loss bound. [Failure counts and source-level passes](heldout_decisions.csv)

## Fixed half-width outages

Here both methods start with the same energy-ranked half of the inputs, keep the same width and use up to two swaps. One arm minimizes the worst error over an intact receiver and all single retained-input outages. The matched arm minimizes their average. The loss denominator is the outer squared loss of the constant fit-row reference. Each outage replaces one selected feature's shape with its fit-row mean for every test observation.

| Classification scope | Δworst normalized squared loss, min − mean | Δintact AUC | Δintact accuracy |
|---|---:|---:|---:|
| Original | −0.941 [−2.829, +0.021] | −0.0266 [−0.0723, +0.0021] | −0.0246 [−0.0841, +0.0078] |
| Wider | +0.530 [−0.019, +1.549] | −0.0207 [−0.0415, −0.0002] | −0.0144 [−0.0222, 0.0000] |
| Pooled | −0.311 [−1.612, +0.665] | −0.0241 [−0.0512, −0.0037] | −0.0202 [−0.0541, +0.0021] |

Retina supplied −3.775 of the four source-level loss contrasts underlying the original mean. On the wider panel, RobotLP supplied +1.549. Single-input minimax passed the loss and both intact quality criteria on Anuran and Dermatology, two of seven sources. It missed the joint criterion on 28/35 classification splits (80%, 57–97%). Relative to the full receiver its worst loss decreased on both panels (−1.009 and −0.107), alongside intact AUC losses of 0.0198 and 0.0148. Against intact-loss swaps the worst-loss differences reversed across panels (−0.950 and +0.525). Regression was secondary. The pooled worst-loss change vs mean was +0.009 [−0.064, +0.106] and scaled MAE increased 0.032.

Equal group weighting changed the Anuran worst-loss contrast from −0.0318 to −0.0078. Its clean AUC/accuracy differences moved from −0.0031/+0.0057 to −0.0067/+0.0010, still inside the margins. Group weighting kept the corresponding Mice and CT-location differences unchanged to four decimals. Group-weighted and row-weighted scores appear separately in the split file.

The evidence supports compression of the full spline and a narrower average support than energy ranking on the classification panels. It does not establish an advantage over matched greedy refitting, or a reliable robustness benefit from optimizing the worst outage. Both primary failures belong in the record of practical comparisons.
