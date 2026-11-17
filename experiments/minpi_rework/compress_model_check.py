"""Check export fidelity using synthetic training and held-aside unlabeled features."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from xgboost import DMatrix

from experiments.minpi_rework.compress_model import CompactForest, fit_compressor
from experiments.minpi_rework.quota_compress_tree import as_scores, extract
from minppi_mofs.modeling import make_model


def main():
    rng = np.random.default_rng(809)
    x = rng.normal(size=(210, 7)).astype(np.float32)
    x[::7, 1] = np.nan
    x[:, -2:] = 0.  # both have zero gain and identical deletion cost
    signal = x[:, 0] + np.nan_to_num(x[:, 1]) + x[:, 2] * x[:, 3]
    tr, cal, query = x[:120], x[120:165], x[165:]
    feature_names = tuple(f"sensor {j}" for j in range(x.shape[1]))
    cases = [("regression", signal, .35), ("binary", np.where(signal > 0, "yes", "no"), .35),
             ("multiclass", np.digitize(signal, [-.6, .6]) + 10, .35), ("regression", signal * 0, .2)]
    with TemporaryDirectory() as tmp:
        for case, (problem, target, rho) in enumerate(cases):
            task = "regression" if problem == "regression" else "classification"
            y = target[:len(tr)]
            compact, report = fit_compressor(tr, y, cal, task=task, feature_names=feature_names, rho=rho,
                                             n_estimators=12, max_depth=3)
            path = Path(tmp) / f"{case}.npz"
            compact.save(path)
            restored = CompactForest.load(path)
            assert restored.names == tuple(feature_names[j] for j in restored.support)
            assert report["deletion_order"].index(5) < report["deletion_order"].index(6)
            teacher = make_model(task=problem, backend="cpu", seed=42,
                                 model_params=dict(n_estimators=12, max_depth=3, learning_rate=.1, n_jobs=1))
            teacher.fit(tr, y if task == "regression" else np.unique(y, return_inverse=True)[1])
            forest = extract(teacher, cal[:1])
            allcols = np.arange(x.shape[1])
            np.testing.assert_allclose(forest.predict(query, allcols),
                                       teacher.get_booster().predict(DMatrix(query), output_margin=True,
                                                                     strict_shape=True), atol=2e-6)
            full_model = CompactForest.from_forest(forest, allcols, feature_names, compact.classes)
            root_splits = np.flatnonzero(forest.left[:, 0] >= 0)
            if len(root_splits):
                t = root_splits[0]
                j = forest.feature[t, 0]
                probe = np.repeat(query[:1], 3, axis=0)
                probe[:, j] = [forest.cut[t, 0], np.nextafter(forest.cut[t, 0], -np.float32(np.inf)), np.nan]
                np.testing.assert_allclose(full_model.predict_margin(probe),
                                           teacher.get_booster().predict(DMatrix(probe), output_margin=True,
                                                                         strict_shape=True), atol=2e-6)
            keep = restored.support
            restricted_x = query[:, keep]
            raw = forest.predict(restricted_x, keep)
            np.testing.assert_allclose(restored.predict_margin(restricted_x), raw, atol=1e-12)
            if task == "regression":
                np.testing.assert_allclose(restored.predict(restricted_x), raw[:, 0], atol=1e-12)
            else:
                scores = as_scores(raw, task)
                np.testing.assert_allclose(restored.predict_proba(restricted_x), scores, atol=1e-12)
                np.testing.assert_array_equal(restored.predict(restricted_x), restored.classes[scores.argmax(axis=1)])
            if case == 3:
                assert len(keep) == 0 and len(restored.feature) == len(restored.roots)
                assert restored.predict(np.empty((3, 0))).shape == (3,)
            print(f"{problem}: retained {len(keep)}/{len(allcols)}, saved {len(restored.value)} nodes; verified")


if __name__ == "__main__":
    main()
