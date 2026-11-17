# Reconstruct the prospective five-source comparison

The blind selection run produced 375 receipts and prediction files for five data sources, five seeds, five selectors and three timing repetitions. The full-path Gain arm also stored the forecast for its narrowest prefix within 0.005 of the best validation score. Each file contains outer row indices. Threshold sketches and model fitting never require outer outcomes.

[Source medians](sources.csv) and [per-seed measurements](seeds.csv) report seconds from the first model fit through deployment prediction, along with supports, test AUC and accuracy, or test R² and MAE. [The receipt index](receipts.csv) lists the hashes checked before scoring. The archived comparator chooses its argmax with ties going to the narrower prefix. The matching-slack comparator shares the exhaustive selection stage and charges its own measured deployment fit.

Run this from the thesis checkout to rebuild the local archive and to verify the tables in a fresh extraction using the five pinned datasets.

```powershell
.\thesis-python.ps1 -m experiments.anchored_vote.package
```

The [prospective replication ZIP](../../../packages/minpi-anchored-prospective.zip) contains the scoring code, source definitions, held-out row indices within the public datasets, original model predictions, receipt hashes and tables. The raw UCI archives can be fetched from their recorded sources. The extracted scorer requires NumPy, pandas and scikit-learn. The paper's experiments and sources can be rebuilt from the full thesis checkout.
