import numpy as np
import pandas as pd

from minppi_mofs.baselines.shap_top_k import fitted_tree_shap_importance


def test_fitted_multiclass_tree_shap_uses_absolute_contributions_excluding_bias():
    class Booster:
        def predict(self, matrix, *, pred_contribs, approx_contribs, strict_shape):
            assert pred_contribs and strict_shape and not approx_contribs
            assert matrix.num_row() == 4 and list(matrix.feature_names) == ["f0", "f1"]
            values = np.array([[-2, 1, 1000], [2, -5, 2000]], dtype=float)
            return np.broadcast_to(values, (4, 2, 3))

    class FittedModel:
        def get_booster(self):
            return Booster()

    frame = pd.DataFrame(np.arange(18).reshape(9, 2), columns=["f0", "f1"])
    np.testing.assert_array_equal(fitted_tree_shap_importance(FittedModel(), frame, seed=42, max_rows=4), [2, 3])
