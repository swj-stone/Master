# Fit and deploy a supervised MinPI random forest

Pass `--reachable-only` to `fit` to export the smallest schema containing every split the selected forest can still reach. A feature appearing only below an excluded split disappears from both the artifact's required input names and its median vector. The output probabilities or regression predictions are identical under every additional availability mask.

To select entire devices or tests, supply `--groups devices.csv` when fitting. Its header is `feature,unit`; include each input column exactly once and use the same unit name for channels acquired together. The MDI baseline sums importance within each unit, and MinPI votes at each possible number of available units. The reported model metadata maps its actual columns to their units. For prediction, `--unit-availability observed.csv` reads 0/1 flags with one column per requested unit, aligned in row order with the prediction inputs. These flags can therefore disable all channels of an absent device together.

Run these commands in the thesis checkout. Fitting uses the supervised two-path selector from the paper, with 100 depth-four trees, seed 42 by default, and an acceptance quota of 0.02. Supply three disjoint labeled splits. Create those splits at the subject or time level when the sampling design requires it. The fitting sample trains the forest and median imputer, the voting sample ranks features, and the acceptance sample selects the width and winning path. The exported forest is the fitted forest truncated to the selected features.

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_rf_deploy fit --fit fit.csv --vote vote.csv --accept acceptance.csv --target outcome --task classification --seed 42 --model selected.npz
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_rf_deploy inspect --model selected.npz
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_rf_deploy predict --model selected.npz --input new_rows.csv --output predictions.csv
```

All three UTF-8 training CSVs must contain the same named numeric features and a target. Reordering columns is allowed. Class labels are read as text, including leading zeros and literal `NA`. All fitting classes must occur in the acceptance split to evaluate macro one-versus-rest AUC. For a continuous target use `--task regression`. Exclude identifiers and other non-predictors from the training schema.

`inspect` prints the winning path, selected feature names in deployment order, and class order. The prediction CSV can contain the full feature schema or only the selected columns, in any order. The output preserves row order. Classification writes the predicted class and one probability per class, with columns named `probability_<label>`. Regression writes the original-scale prediction.

A missing numeric observation in an available column uses that feature’s fitting median. To declare a sensor physically unavailable for particular rows, supply `--availability availability.csv`. This file contains a 0 or 1 for each selected feature, in the same row order as the prediction CSV. A tree stops at the first unavailable split and returns the training-weighted prediction stored at that node. A selected column can be omitted entirely from the prediction CSV when its availability column is zero for every row. For an empty support, an arbitrary row-ID field in the prediction CSV preserves its row count.

The artifact stores selected feature medians, class labels, thresholds, child indices, and prediction values for reachable tree nodes in compressed NumPy arrays. Loading disables pickle and validates the tree graph and array shapes. For a minimal service, copy `src/minppi_mofs/predictive_rf_artifact.py` alongside the application and install NumPy. Standalone inference on an array uses

```python
from predictive_rf_artifact import RestrictedRF

model = RestrictedRF.load("selected.npz")
outputs = model.predict(x_selected, available=availability_bool)
```

Here `x_selected` follows `model.names`; omit `available` for an intact panel. Classification returns an array of shape `(n_rows, n_classes)` in `model.classes` order.

To fit a registered binary classification example from locally prepared source data and compare both paths against signed forecasts for Votes, German Credit and Beijing at seed 42, run

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_rf_deploy fit --dataset votes --seed 42 --model selected-votes.npz
.\thesis-python.ps1 -m experiments.minpi_rework.predictive_rf_deploy_check --sealed
```

Registered datasets use identifiers `f0`, `f1`, … in the provider's column order and its encoded class labels. The portable check without `--sealed` uses generated multiclass and regression examples and requires no external data.
