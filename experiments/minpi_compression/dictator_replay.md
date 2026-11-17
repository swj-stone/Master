# Reusing votes from adjacent coalitions

This retrospective analysis was designed after the main confirmation scores were opened. Its eight sources and 16 wider folds are the ones already reported. The wider replay uses the same capped inputs, splits and sampler seeds as `wide_transfer.py`. It never opens those outer labels. It records hashes of their out-of-sample forecasts and compares them directly.

The frozen local minimum, coefficient rank, a cached implementation of the local minimum, its cross-coalition dictator bound, and the sharp closure of the final cache all pass through the same original inner width rule. No method gets another validation threshold. An observed winning coalition excludes absent features as possible dictators, and a losing coalition excludes present features. Once a feature is excluded, its upper bound is at most zero. The online algorithm still checks all scheduled winning bases for each feature lacking a negative witness. This preserves the first scheduled negative edge.

The final closure additionally reads each observed losing coalition and looks for its observed winning parents. Such an edge certifies an exact negative, including when its two endpoints were evaluated for different features. For every feature with no downward edge, the largest possible exact minimum over unrestricted Boolean completions is +1 if all observed votes agree with membership of that feature and zero otherwise. Sharpness here concerns Boolean completions. A particular ridge design can impose additional restrictions.

The ranking clocks include preprocessing. Total selection clocks also include the common held-out width selection. Each split rotates a random ordering of five methods five times, placing every method at every timing position. Timing aggregates use the median on each split and geometric paired ratios, equally weighting sources. The `cached` control isolates the cost of the cross-coalition rule from that of caching. The per-split published votes are checked against the independent SVD oracle wherever it was feasible.

```powershell
.\thesis-python.ps1 -m experiments.minpi_compression.dictator_audit
.\thesis-python.ps1 -m experiments.minpi_compression.dictator_summary
.\thesis-python.ps1 -m pytest tests/test_dictator_minpi.py -q
```

The tables are under `thesis-paper/evidence/minpi_compression/dictator/`. The per-feature table records the first scheduled witness and a witness from the sharp closure. The run table records every clock, support, and forecast hash. Source and domain contrasts give held-out prediction differences on the eight-source panel. Raw MAE is reported within each source. Domain contrasts use standardized MAE.
