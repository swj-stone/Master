# One guarded deletion from a coefficient-selected model

The [fixed retrospective protocol](../../../../experiments/minpi_compression/witness_guard.md) covers all twelve previously scored UCI sources and slices from both panels. The three policies start with exactly the same coefficient-selected prefix. Each incurs the MinPI voting cost to determine how many members have a witnessed negative edge. They differ only in which members they propose to drop. Each policy may spend one additional fit on the width-validation split. An unguarded deletion would answer a different deployment question.

| Relative to the standalone coefficient selector | Classification, 6 sources | Regression, 6 sources |
|---|---:|---:|
| Splits with an eligible deletion / 30 | 9 | 7 |
| Accepted proposals, witness / tail / uniform | 4 / 3 / 3 | 5 / 0 / 3 |
| Witness retained width ratio | .9833 | .9437 |
| Source-bootstrap 95% interval for width ratio | [.9613, 1.0000] | [.8733, 1.0000] |
| Change in AUC / accuracy | −.0016 / +.0003 | — |
| Change in R² / standardized MAE | — | +.0005 / −.0047 |
| Witness selection time ratio | 1.1425 | 1.8293 |

Voting identified 49 feature/split instances inside the selected prefix. The guard approved nine deletions totaling 16 feature/split instances on Glass, Energy, Biodegradation and Aquatic Toxicity. Each matched arm incurred 2,313 ridge factorizations during voting, 315 fits for choosing prefix width, 16 optional fits and 60 final fits over the sixty splits. The coefficient selector used 60 ranking factorizations, 315 width fits and 60 final fits. Verified negative edges and accepted model changes measure different events.

Neither domain reached 10% further compression over coefficient selection. Both domain-level predictive changes relative to coefficient ranking were within the original margins. Against full ridge the classification accuracy fell by 1.59 percentage points; the margin was one point. Witness and matched uniform deletion gave the same final support on 54 of 60 splits. On the regression panel the five accepted witness proposals compare with zero for the coefficient-tail control, but three for uniform deletion. One random draw per split and repeated observations of the same sources constrain the interpretation. The [source-resampling intervals](paired-source-intervals.csv) use six entire sources per domain with all related seeds kept together.

In [scores.csv](scores.csv), `h` counts witnessed features inside the coefficient prefix, `optional_refits` counts attempted proposals, and `accepted` records actual deployment of a smaller model. `width_fits` excludes the ridge factorizations during ranking, the optional refit and the final deployment fit. The matched controls compute the same votes to obtain the same count. Consequently their times reflect an experimental control, whereas the standalone coefficient row measures the cost of the cheap alternative on its own. [Per-source means](sources.csv) · [Paired domain contrasts](effects.csv) · [Independently replayed guards](guard-audit.csv)

## Replay

Place the fingerprinted data from both source panels in the thesis data directory, following their acquisition guides. The two panels' published forecast archives are used for the exact identity check of the common coefficient baseline. Run the following commands from the thesis checkout.

```powershell
.\thesis-python.ps1 -m experiments.minpi_compression.witness_guard select
.\thesis-python.ps1 -m experiments.minpi_compression.witness_guard score
.\thesis-python.ps1 -m experiments.minpi_compression.witness_guard_audit
```

The [forecast archive](forecasts.zip) contains every split's predictions and selection records along with their hashes. To rescore it with the recorded code, extract it into `results/minpi_compression/witness-guard-r1/` and run the last two commands. Wall times use the per-method median of three reruns within each split. Scoring independently refits each selected model with an SVD solver. The final audit checks every negative edge used for deletion against directly fitted ridge models and replays each optional validation decision.
