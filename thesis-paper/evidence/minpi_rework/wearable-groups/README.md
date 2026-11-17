# Acquisition units in two subject-held-out activity studies

**The frozen MinPI forest retired zero physical devices in 18 validation folds.** An exhaustive search across every subset of the same frozen forest gave the same result. On the five-device sports data, six of eight folds had a one-device deletion that passed the intact accuracy and AUC gates. Every four-device frozen set failed the outage quota. The smallest attainable worst excess at that width ranged from .030 to .051 against the .020 allowance. On MHEALTH the corresponding range was .231–.306. This limitation applies even when subset search is exact.

Table entries are means over held participants, with each participant weighted once. “Joint on held” counts subjects meeting both intact performance floors and the uniform-device outage quota relative to the same full forest. These are evaluations on unseen subjects after selecting the model on separate acceptance participants.

| Dataset | Method | Mean devices | Folds retiring ≥1 | Macro AUC | Accuracy | Joint on held |
|---|---|---:|---:|---:|---:|---:|
| MHEALTH, 3 devices | Full, MDI, MinPI, portfolio, exhaustive frozen | 3.0 | 0/10 | .9978 | .9268 | 10/10 |
| | Frozen, force one retirement | 2.0 | 10/10 | .9963 | .8841 | 0/10 |
| | Exhaustive refits, joint guard | 2.1 | 8/10 | .9952 | .8950 | 6/10 |
| | Exhaustive refits, intact guard | 2.0 | 8/10 | .9927 | .8867 | 6/10 |
| Daily and Sports Activities, 5 devices | Full, MDI, MinPI, portfolio, exhaustive frozen | 5.0 | 0/8 | .9840 | .8000 | 8/8 |
| | Frozen, force one retirement | 4.0 | 8/8 | .9817 | .7913 | 0/8 |
| | Exhaustive refits, joint guard | 1.5 | 8/8 | .9818 | .7913 | 3/8 |
| | Exhaustive refits, intact guard | 1.5 | 8/8 | .9822 | .7866 | 3/8 |

At one forced retirement, the two rankings and exhaustive frozen search all removed the chest unit on MHEALTH and the right arm unit on the sports source. All 18 forced supports failed their joint acceptance gate. The average decline in held accuracy was 4.27 percentage points and 0.87 points, respectively. Refitting all subsets made retirement feasible on the acceptance subjects, but the joint guard transferred to only nine of the eighteen held subjects. Across participants, the test sets contain 1,251 steady MHEALTH windows and 9,120 sports windows. Every held subject had examples of every target class. Training folds overlap, so the windows are unsuitable as independent replicates.

For the frozen selectors, three warmed complete fits per subject gave source median fit-to-forecast times of .214 s for the portfolio and .199 s for MDI alone on MHEALTH, and 1.677 s versus 1.436 s on the sports source. Exhaustive supervised evaluation required six and thirty additional forest fits per fold; median additional refit-and-acceptance times were .864 s and 20.977 s. The full forest and empirical class prior supplied the remaining two candidates. Native sklearn traversal of every held input under every possible availability mask reproduced the stored probability vectors within 8.9 × 10⁻¹⁶ across all 336 fold-mask combinations.

Uniform random device availability at each count is a defined failure stress for devices with documented physical locations. Neither source measures independent failure rates, device acquisition prices, or deployment power. MHEALTH evaluation excludes windows that contain transitions or the null label. The sports archive uses scripted activities and fixed five-second windows. The two small volunteer cohorts are insufficient to generalize to unconstrained activity recognition.

## Sources and reproduction

The registered study is commit `b6bfab2`. Its acquisition maps and raw archive digests are recorded in `experiments/minpi_rework/wearable_sources.json`; the data provider defines the feature order. All 18 selection hashes were frozen in `88278c3` before held scoring. The forecast archive digest is in `archive-sha256.txt`. `held.csv` contains every subject and method, `acceptance-subsets.csv` contains the complete support search, and `native-checks.csv` and `selector-runtime.csv` contain native prediction and clock audits. [MHEALTH, Banos, Garcia and Saez](https://doi.org/10.24432/C5TW22). [Daily and Sports Activities, Altun, Barshan and Tunçel](https://doi.org/10.24432/C5C59F).

To reproduce the original scores offline, place the signed forecast ZIP and both checksum-pinned prepared NPZ files in the indicated paths. The output directory receives independently regenerated CSV files.

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.wearable_score --prepared data/minpi_rework --archive packages/minpi-wearable-forecasts.zip --output results/minpi_rework/wearable-rescore
.\thesis-python.ps1 -m experiments.minpi_rework.wearable_audit --prepared data/minpi_rework --archive packages/minpi-wearable-forecasts.zip --output results/minpi_rework/wearable-runtime-replay
```

## Follow-up with path-nominated refits

After seeing the frozen study outcomes, we registered a comparison of nominated refits in `21d9769` and sealed its forecasts in `ca12510`. The study reused the same subjects. Three rankings nominated one support at each width, using impurity, minimum or pooled size votes. The one-step alternatives trained only the proposed one-device retirement and returned the full forest on rejection. All methods used the same acceptance rules and retrained their nominated forests from the original fitting subjects. Clocks include selection and intact prediction, with one interleaved complete run per arm and held participant. Fits include the full reference forest. The joint column counts held participants satisfying both the outage quota and the intact floors.

| Dataset | Nominations | Mean devices | Mean fits | Median s | AUC | Accuracy | Joint |
|---|---|---:|---:|---:|---:|---:|---:|
| MHEALTH | One MDI retirement | 2.30 | 2 | .343 | .9973 | .9235 | 7/10 |
| | MDI path | 2.20 | 3 | .495 | .9957 | .9056 | 6/10 |
| | Minimum path | 2.30 | 3 | .508 | .9973 | .9235 | 7/10 |
| | Pooled path | 2.40 | 3 | .508 | .9974 | .9226 | 8/10 |
| | MDI + minimum paths | 2.20 | 3.40 | .514 | .9957 | .9056 | 6/10 |
| | Exhaustive refits | 2.10 | 7 | 1.080 | .9952 | .8950 | 6/10 |
| Daily and Sports Activities | One MDI retirement | 4.625 | 2 | 1.940 | .9863 | .8213 | 8/8 |
| | MDI path | 3.50 | 5 | 3.871 | .9817 | .8104 | 7/8 |
| | Minimum path | 3.50 | 5 | 4.090 | .9817 | .8104 | 7/8 |
| | Pooled path | 3.50 | 5 | 4.094 | .9817 | .8104 | 7/8 |
| | MDI + minimum paths | 3.50 | 5.125 | 4.096 | .9817 | .8104 | 7/8 |
| | Exhaustive refits | 1.50 | 31 | 22.431 | .9818 | .7913 | 3/8 |

On MHEALTH the single MDI retirement and the entire minimum path selected identical supports and forecasts for all ten held subjects. On the sports data the MDI, minimum and pooled paths selected identical supports in all eight folds. Adding the minimum path to MDI changed no selected support across either dataset. In the sports source the path methods saved about 18 s per fold against exhaustive refitting and retained two additional devices on average. Exhaustive search minimized width on acceptance participants. The minimum vote supplied no observed advantage beyond the cheaper impurity-based control on either physical acquisition task.

All nine arms and all subject-level measurements are in `refit-held.csv` and `refit-summary.csv`. Their deterministic offline replay uses the original data summaries and both sealed forecast archives.

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.wearable_refit_offline --prepared data/minpi_rework --archive packages/minpi-wearable-forecasts.zip --refit-archive packages/minpi-wearable-refit-forecasts.zip --output results/minpi_rework/wearable-refit-rescore
```
