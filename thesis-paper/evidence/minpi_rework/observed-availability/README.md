# PM2.5 prediction during recorded Beijing monitor outages

Across 12 stations, 3,394 noon origins in the final 292-day block had observed PM2.5 concentrations the following noon. Of these, 239 station-days spanning 140 dates had missing predictors. There were 474 absent encoded coordinates, including two for any missing wind direction. The original archive contained 17,532 noon origins. We excluded 605 without an observed next-day target and 35 additional station-days whose response crossed a split boundary. This is one time series of a station network. The earlier Beijing experiment used one of its stations. [Protocol](../../../../experiments/minpi_rework/observed_availability.md)

All five ranking and acceptance combinations retained PM10, atmospheric pressure, dew point, wind speed and the cosine of wind direction. Ordinary MDI with the acceptance rule returned the same columns as both MinPI portfolios. Zeroing each row's unreachable path increments gave conditional excess losses within 4.1 × 10⁻¹² of direct traversal over all 4,096 further masks in the three-row check.

| Outer station-days | Deployed columns | Missing-input handling | Mean prediction | MAE | MSE | R² |
|---|---:|---|---:|---:|---:|---:|
| All 3,394 | 12 | Fit median | 78.74 | 50.39 | 4,912.18 | .1961 |
| All 3,394 | 12 | Stop | 78.98 | 50.53 | 4,925.92 | .1938 |
| All 3,394 | 5 | Fit median | 79.52 | 51.24 | 4,989.96 | .1834 |
| All 3,394 | 5 | Stop | 79.76 | 51.37 | 5,001.03 | .1816 |
| Incomplete 239 | 12 | Fit median | 75.20 | 49.09 | 4,232.10 | .2331 |
| Incomplete 239 | 12 | Stop | 78.64 | 51.12 | 4,427.20 | .1978 |
| Incomplete 239 | 5 | Fit median | 76.27 | 49.58 | 4,269.38 | .2264 |
| Incomplete 239 | 5 | Stop | 79.58 | 51.39 | 4,426.55 | .1979 |

The mean observed PM2.5 was 73.70 across the entire holdout and 74.54 on the incomplete subset. Among incomplete rows, stopping at the first absent measurement increased MAE by 2.03 (paired 95% week-block interval 0.81 to 3.28) with the full forest and 1.81 (0.61 to 2.97) with the selected columns. Under stopping, compressing to five columns increased MAE by 0.27 (−0.15 to 0.63) in this subset and by 0.83 (0.54 to 1.13) overall. Both inference policies gave identical predictions on the 3,155 complete rows at each fixed support. These intervals characterize daily variability in the observed network conditional on a recorded future outcome.

The frozen [selections and counts](lock.json), [per-station-day forecasts](forecasts.csv), [full score table](scores.csv) and [paired changes](paired.csv) are available. Run `python -m experiments.minpi_rework.observed_availability score` to rescore the existing forecasts. Use `collect` in a separate extraction to refit and repeat the all-mask computation.
