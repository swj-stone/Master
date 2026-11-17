# Original six prepared datasets, five seeds

The first three columns use the same fitted depth-four random forest and original test rows. MDI and MinPI share the acceptance quota and the intact-metric check. MinPI can choose from the MDI and minimum-vote paths. The archival XGBoost pipeline ranks by gain and refits on the selected columns using the full development sample. It answers the practical question of what can be achieved if replacing the original model is allowed.

Each cell gives mean retained columns · AUC / accuracy for classification or R² / MAE for Abalone · fit-inclusive seconds. Means use seeds 42, 123, 456, 789 and 1024. The two learners were timed in different study runs; treat differences in seconds accordingly. Historical preprocessing retains different widths on some Spambase splits.

| Source | Full RF | RF MDI | RF MinPI | XGB Gain refit |
|---|---|---|---|---|
| abalone | 8.0 · 0.4968 / 1.6086 · 0.25s | 2.6 · 0.4811 / 1.6491 · 0.25s | 2.6 · 0.4811 / 1.6491 · 0.31s | 2.4 · 0.5225 / 1.5638 · 0.94s |
| credit card fraud | 29.0 · 0.9580 / 0.9993 · 25.74s | 19.2 · 0.9561 / 0.9993 · 28.76s | 17.8 · 0.9554 / 0.9993 · 31.85s | 10.8 · 0.9711 / 0.9994 · 4.74s |
| gisette | 100.0 · 0.9719 / 0.9191 · 0.32s | 67.6 · 0.9688 / 0.9124 · 0.81s | 64.6 · 0.9682 / 0.9123 · 1.37s | 35.0 · 0.9849 / 0.9521 · 1.30s |
| isolet | 100.0 · 0.9765 / 0.6853 · 1.13s | 79.0 · 0.9759 / 0.6806 · 5.69s | 78.6 · 0.9758 / 0.6797 · 10.28s | 75.0 · 0.9928 / 0.8269 · 34.90s |
| madelon | 100.0 · 0.7277 / 0.6735 · 0.27s | 53.6 · 0.7236 / 0.6673 · 0.64s | 40.0 · 0.7240 / 0.6723 · 1.06s | 20.4 · 0.8302 / 0.7585 · 1.24s |
| spambase | 55.8 · 0.9704 / 0.9216 · 0.19s | 26.4 · 0.9688 / 0.9188 · 0.47s | 25.8 · 0.9686 / 0.9192 · 0.79s | 25.2 · 0.9840 / 0.9459 · 1.19s |

The archived refit uses fewer columns on all six sources and has higher mean AUC on all five classification sources and lower MAE on Abalone. In the matched frozen-forest comparison, Madelon accounts for the largest width reduction.

[Forest scores](scores.csv) · [Archived refit on the primary test rows](../quota-tree/historical-42-123-456-789-1024.csv) · [Numeric table](refit-comparison.csv)
