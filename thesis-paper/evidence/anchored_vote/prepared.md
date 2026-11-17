# Anchored MinPI on the original six prepared sources

Use the independently copied, hash-checked prepared splits in the thesis checkout. The five original seeds are 42, 123, 456, 789 and 1024. The source panel has been used in development. Its 30 comparisons describe continuity with previous work.

The R3 experiment measures the minimum, pooled mean and Gain-only polishers with the same inner anchor, two refinement opportunities and native XGBoost policy. Arm order rotates across source and seed. Each source–seed–arm has one measured CUDA run from full fit through final refit and prediction. Source reads, shared synthetic CUDA warm-up, output writes and scoring are outside this interval. The exhaustive and geometric Gain controls come from the immediately preceding R2 session on the same device, with the same prepared splits, hyperparameters, utilities and native final refit. Between-session system load can affect their timing. The report checks each anchor's exact column set and validation scores against the full exhaustive curve, as well as the common executable model helper's digest. The geometric and exhaustive Gain paths use different width objectives; the exhaustive near-best width sensitivity is included.

| Source | MinPI columns | Direct Gain columns | MinPI seconds | Direct Gain seconds | MinPI AUC or R² | Direct Gain AUC or R² |
|---|---:|---:|---:|---:|---:|---:|
| Abalone (R²) | 5 | 7 | 0.91 | 1.33 | .54635 | .54635 |
| Madelon | 7 | 7 | 1.04 | 16.16 | .87197 | .87078 |
| Spambase | 17 | 45 | 0.98 | 8.76 | .98547 | .98637 |
| Gisette | 26 | 77 | 1.08 | 17.22 | .98604 | .98862 |
| Credit Card Fraud | 11 | 26 | 3.06 | 16.89 | .97429 | .97717 |
| ISOLET | 24 | 92 | 23.55 | 354.68 | .98936 | .99355 |

Entries are medians over the original five seeds. All six sources passed the paired-median joint target and the 0.01 quality guard against exhaustive argmax Gain. Of 30 individual pairs, 27 passed on time and width; 26 also passed on held score. The current Gain rerun exactly reproduced all 30 original Gain orders, selected sets and test prediction arrays. Against geometric Gain, 5/6 sources and 19/30 individual pairs met all three conditions. Against the equally anchored Gain-only method in the same session, the corresponding counts were 0/6 and 2/30. The lowest-vote rule differed in final support from the mean-vote rule in 4/30 cases.

At the same 0.005 validation allowance, MinPI passed the width and predictive comparison against exhaustive Gain on 3/6 sources. MinPI exceeded the paired width allowance on Credit Card Fraud, Spambase and ISOLET. On ISOLET both marginal median widths are 24 but the median paired width excess, after subtracting the allowed increase seed by seed, is +1.95 columns. [Per-seed results](prepared-seeds.csv), [paired comparisons](prepared-sources.csv), [common-tolerance deployments](prepared-common-slack.csv) and the [exhaustive replay](../minpi_replication/common-slack-development.csv) contain all outcomes.

From the thesis checkout, run these commands sequentially with the shared virtual environment:

```powershell
.\thesis-python.ps1 -m experiments.split_vote.run --tag six-paired --seeds 42 123 456 789 1024 --methods min_mass gain_grid exhaustive
.\thesis-python.ps1 -m experiments.minpi_replication.common_slack freeze development
.\thesis-python.ps1 -m experiments.minpi_replication.common_slack score development
.\thesis-python.ps1 -m experiments.anchored_vote.run --names abalone madelon spambase gisette credit_card_fraud isolet --seeds 42 123 456 789 1024 --tag original-r3 --device cuda --arms min mean gain
.\thesis-python.ps1 -m experiments.anchored_vote.prepared_report
```

The report validates all 150 paired-arm forecast receipts before accessing the test responses. It computes ordinary held-out AUC and accuracy or uncensored R² and raw MAE. Paired seed-level wall-time and width differences govern source decisions. The permitted width increase is the larger of 1 column and 5% of that seed's comparator width. The companion score gate allows 0.01 of AUC or R² loss. Source tables also show how many seeds actually satisfy all conditions simultaneously.

`prepared-common-slack.csv` compares exhaustive Gain and minimum voting when each can trade 0.005 validation utility for compression. It uses measured test predictions at the corresponding selected widths. For exhaustive Gain the 0.005 support sometimes differed from its argmax, and the additional final fits did not capture the exhaustive end-to-end time with that decision rule.
