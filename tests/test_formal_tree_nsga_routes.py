"""Exercise both standalone selectors through the formal scoring and fit-accounting boundary."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments import literature_worker as worker
from minppi_mofs.baselines import nsga2_fs, shap_top_k


@pytest.mark.parametrize("method", ["tree_shap_top_k", "nsga2"])
def test_standalone_matches_reference_and_counts_fits(method, tmp_path, monkeypatch):
    source = tmp_path/"inputs"/"demo"
    source.mkdir(parents=True)
    seed = 73
    rng = np.random.default_rng(seed)
    cols = ["a[0]", "b", "c", "d"]
    arrays = {}
    for role, size in (("train", 112), ("val", 84), ("test", 56)):
        x = pd.DataFrame(rng.normal(size=(size, len(cols))).astype("float32"), columns=cols)
        arrays[role] = x, (x.iloc[:, 0] + x.iloc[:, 1]/3 > 0).astype(int).to_numpy()
    policy = dict(task="binary", backend="cpu", seed=seed,
                  model_params=dict(n_estimators=4, max_depth=2, n_jobs=1))
    meta = dict(features=cols, utility="roc_auc", policy=policy,
                sha256={"test.npy":"synthetic", "test_y.npy":"synthetic"})
    (source/"metadata.json").write_text(json.dumps(meta))
    out = tmp_path/"result"

    def load(_, dataset, role):
        assert dataset == "demo"
        if role == "test":
            assert (out/"frozen.json").exists()
        return arrays[role]

    monkeypatch.setattr(worker, "load_split", load)
    result = worker.trial("demo", "shared", method, out, seed=seed, input_root=source.parent, enforce_guard=False)
    estimate = json.loads((out/"estimator.json").read_bytes())
    fits = [json.loads(line) for line in (out/"fits.jsonl").read_text().splitlines()]
    completed = [event for event in fits if event["event"] == "completed"]
    search_fits = sum(event["phase"] == "coalition" for event in completed)
    assert result["fits"] == len(completed)
    assert result["selection_fits"] == search_fits
    x, y = arrays["train"]
    if method == "tree_shap_top_k":
        assert search_fits == 0
        scores = shap_top_k.compute_shap_topk_importance(x, y, **policy)
        np.testing.assert_allclose(estimate["scores"], list(scores.values()), rtol=1e-6)
        assert estimate["explained_rows"] == len(x)
    else:
        xv, yv = arrays["val"]
        with np.load(out/"predictions.npz") as predictions:
            rank = predictions["rank_rows"]
        front = nsga2_fs.nsga2_feature_selection(x, y, xv.iloc[rank], yv[rank], utility="roc_auc", **policy)
        expected = [(np.flatnonzero(mask).tolist(), ratio, loss) for mask, (ratio, loss) in front.pareto_front]
        actual = [(p["indices"],p["cardinality_ratio"],p["validation_utility_loss"]) for p in estimate["pareto_front"]]
        assert actual == expected
        best = min(expected, key=lambda p:(p[2],p[1],p[0]))
        assert result["selected"] == [cols[i] for i in best[0]]
        assert estimate["logical_evaluations"] == 24 == estimate["unique_subsets"] + estimate["cache_hits"]
        assert search_fits > 0
