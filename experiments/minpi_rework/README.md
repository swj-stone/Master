# Export and use a MinPI-compressed XGBoost ensemble

The `compress_model` command takes a numeric training CSV with a target and a separate calibration CSV with the same feature names. Calibration uses predictors alone. It fits the teacher, runs the three-stratum MinPI greedy deletion path and saves a compressed tree ensemble. When a tree first encounters a removed split, the exporter turns that node into a leaf with its cover-weighted descendant value and omits its descendants. Every surviving split uses the retained column's position in the new input schema. NaN follows the teacher's default branch. The deployed Python predictor requires NumPy; the CSV command also uses pandas.

### Run on the prepared Madelon split

Paste into PowerShell from the project directory. Paths to the model and the two data files are absolute because the Python launcher enters the thesis worktree.

```powershell
Set-Location -LiteralPath 'C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper'
$thesis = (Resolve-Path -LiteralPath '.\MastersThesis').Path
$python = Join-Path $thesis 'thesis-python.ps1'
$split = Join-Path $thesis 'results\icomp2026_formal\icomp2026-formal-v1-s42-123\preprocess\madelon\s42'
$model = Join-Path $thesis 'results\minpi-deploy\madelon-s42.npz'
& $python -m experiments.minpi_rework.compress_model fit --train (Join-Path $split 'train.csv') --calibration (Join-Path $split 'val.csv') --target label --task classification --rho 0.20 --seed 42 --model $model
```

The printed report gives `retained_names`, the number of retained inputs, calibration discrepancy, deletion order and time. The default fits 100 depth-four trees at learning rate .1, with a single CPU worker. `--device cuda` selects GPU training. Published experiment timings and the CLI's separately reported serialization time describe different measurements.

```powershell
$predictions = Join-Path $thesis 'results\minpi-deploy\madelon-predictions.csv'
& $python -m experiments.minpi_rework.compress_model predict --model $model --input (Join-Path $split 'val.csv') --output $predictions
```

The prediction CSV contains the predicted class and one probability per class, or the predicted numeric value for `--task regression`. Input can contain **only the retained names** in any order. In the example, the reader selects those columns from the original calibration CSV. This checks application of the exported model on familiar rows. Measuring generalization requires separate untouched observations and labels.

For another dataset, give `--train` and `--calibration` files from disjoint row or subject partitions. Use `--row-id subject_id` when both CSVs contain a stable ID. The command then rejects shared IDs and excludes that column from fitting. Otherwise establish disjointness upstream. All training columns except the target and optional ID are numeric predictors. Additional columns in calibration or prediction CSVs are ignored. Supply the same preprocessing and measurement units at deployment. The scalar `rho` bounds the calibration margin discrepancy relative to the distance between the teacher and its empty-input prediction. It provides no direct bound on accuracy or MAE.

### Predict from a reduced NumPy array

```python
from experiments.minpi_rework.compress_model import CompactForest

model = CompactForest.load(model_path)
print(model.names)    # Required names, in array column order
print(model.support)  # Corresponding zero-based positions in the training CSV's predictors
predictions = model.predict(x_retained)
```

For classification, `model.predict_proba(x_retained)` orders probabilities by `model.classes`. If no input was retained, use a NumPy array of shape `(row_count, 0)` or pass `--rows 25` in place of `--input <path>`. The archive stores the selected schema, class labels, bias and reachable tree nodes. It can be loaded in another process without the training files, the fitted Booster or XGBoost.

Run `& $python -m experiments.minpi_rework.compress_model_check` for deterministic synthetic round-trip checks of regression, binary and multiclass probabilities, exact split-threshold ties, NaN routing, equal-cost unused inputs, and empty-input deployment.
