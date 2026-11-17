# Anchored MinPI on the original six prepared sources

Use the independently copied, hash-checked prepared splits in the thesis checkout. The five original seeds are 42, 123, 456, 789 and 1024. The source panel has been used in development. Its 30 comparisons describe continuity with previous work.

The R3 experiment measures the minimum, pooled mean and Gain-only polishers with the same inner anchor, two refinement opportunities and native XGBoost policy. Arm order rotates across source and seed. Each source–seed–arm has one measured CUDA run from full fit through final refit and prediction. Source reads, shared synthetic CUDA warm-up, output writes and scoring are outside this interval. The exhaustive and geometric Gain controls come from the immediately preceding R2 session on the same device, with the same prepared splits, hyperparameters, utilities and native final refit. Between-session system load can affect their timing. The report checks each anchor's exact column set and validation scores against the full exhaustive curve, as well as the common executable model helper's digest. The geometric and exhaustive Gain paths use different width objectives; the exhaustive near-best width sensitivity is included.

From the thesis checkout, run these commands sequentially with the shared virtual environment:

```powershell
.\thesis-python.ps1 -m experiments.split_vote.run --tag six-paired --seeds 42 123 456 789 1024 --methods min_mass gain_grid exhaustive
.\thesis-python.ps1 -m experiments.anchored_vote.run --names abalone madelon spambase gisette credit_card_fraud isolet --seeds 42 123 456 789 1024 --tag original-r3 --device cuda --arms min mean gain
.\thesis-python.ps1 -m experiments.anchored_vote.prepared_report
```

The report validates all 150 paired-arm forecast receipts before accessing the test responses. It computes ordinary held-out AUC and accuracy or uncensored R² and raw MAE. Paired seed-level wall-time and width differences govern source decisions. The permitted width increase is the larger of 1 column and 5% of that seed's comparator width. The companion score gate allows 0.01 of AUC or R² loss. Source tables also show how many seeds actually satisfy all conditions simultaneously.
