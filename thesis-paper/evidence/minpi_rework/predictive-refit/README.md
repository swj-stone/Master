# Refitting under the same forest and mask objective

Seven transfer sources use five seeds each. All four wide sources use seed 42 for this secondary test. Every model uses the same training rows and bootstrap recipe as its parent. All 39 jobs sealed their selected supports and prediction hashes before releasing the wide outcomes. Scoring reproduced each sealed forecast. All 56 small-forest size-level checks agreed with direct mask enumeration, including distinct empty predictions.

Averaged metrics below weight sources equally. MAE is divided by the response standard deviation on development rows. Time sums the independently measured components needed by each arm through acceptance. The widths refer to the declared schema. The frozen-arm timing uses the scalar selector in this study, which predates the subsequent interval implementation. The 20% and 50% grids use descending widths under one fixed MDI order. A single optional refit uses the previously accepted width and survives only when it passes both checks and improves the worst-size calibration loss.

| Panel | Method | Columns | Fits | Seconds | AUC / accuracy | R² / scaled MAE | Failed quota |
|---|---|---:|---:|---:|---|---|---:|
| seven | full | 12.14 | 1.0 | 0.088 | 0.8325 / 0.8148 | 0.3170 / 0.5628 | 0 |
| seven | mdi | 7.20 | 1.0 | 0.116 | 0.8282 / 0.8148 | 0.2821 / 0.5848 | 9 |
| seven | mdi_one_refit | 7.20 | 2.0 | 0.231 | 0.8286 / 0.8171 | 0.3025 / 0.5710 | 3 |
| seven | minpi | 6.77 | 1.0 | 0.132 | 0.8271 / 0.8139 | 0.2812 / 0.5869 | 10 |
| seven | minpi_one_refit | 6.77 | 2.0 | 0.246 | 0.8258 / 0.8119 | 0.3011 / 0.5742 | 6 |
| seven | refit20 | 4.83 | 10.1 | 0.908 | 0.8166 / 0.8103 | 0.3099 / 0.5672 | 7 |
| seven | refit50 | 6.69 | 5.0 | 0.436 | 0.8187 / 0.8104 | 0.3084 / 0.5667 | 8 |
| wide | full | 795.25 | 1.0 | 0.356 | 0.9043 / 0.9097 | 0.3345 / 0.7646 | 0 |
| wide | mdi | 129.25 | 1.0 | 0.505 | 0.9045 / 0.9078 | 0.3133 / 0.7767 | 0 |
| wide | mdi_one_refit | 129.25 | 2.0 | 0.881 | 0.9017 / 0.9268 | 0.3441 / 0.7626 | 0 |
| wide | minpi | 120.75 | 1.0 | 2.190 | 0.9015 / 0.9085 | 0.3133 / 0.7767 | 1 |
| wide | minpi_one_refit | 120.75 | 2.0 | 2.565 | 0.9007 / 0.9296 | 0.3441 / 0.7626 | 0 |
| wide | refit20 | 98.50 | 28.0 | 4.892 | 0.8957 / 0.9317 | 0.3939 / 0.7276 | 0 |
| wide | refit50 | 84.75 | 10.5 | 1.993 | 0.9012 / 0.9316 | 0.3953 / 0.7263 | 0 |

On the seven-source panel, the 20% deletion grid gave a smaller model and better average regression scores at the cost of about ten forest fits. The frozen MinPI comparison had higher average AUC and accuracy there. A one-refit MDI alternative passed the held-out risk quota more often than the corresponding MinPI alternative. On the four wider seed-42 fits, the two refit grids both achieved better average regression scores and accuracy than frozen MinPI. Timing and predictive choices depend on whether replacing the original forest is permitted.

[Preregistered comparison](../../../../experiments/minpi_rework/predictive_refit_protocol.md) · [Forecast lock](lock.json) · [All decisions and scores](scores.csv)
