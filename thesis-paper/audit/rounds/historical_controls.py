"""Check the historical source, estimator receipts, selected columns and held-out scores."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error, r2_score
from threadpoolctl import threadpool_limits

from pmlb_baselines import HERE, ROOT, auc, read, sha


RUN = ROOT / "results/thesis_benchmark/baselines-eight-preprocessed-five-seeds"


def check_support(method, meta, saved, x, result):
    cols = np.array(meta["features"])
    info = {}
    if method in {"shapg", "banzhaf", "sampling_shap", "tree_shap_top_k"}:
        order = cols[np.argsort(-np.asarray(saved["scores"]), kind="stable")].tolist()
        sized = saved["sizing"]
        assert order == sized["order"]
        selected = set(order[:sized["k"]])
        if method == "shapg":
            assert saved["depth"] == 1 and saved["max_neighbors"] == 3
            # Recreate Algorithm 1 by global edge order, separate from the
            # experimental code's vectorised strongest-incident-neighbour shortcut.
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.nan_to_num(abs(np.corrcoef(x, rowvar=False)))
            a, b = np.triu_indices(len(cols), k=1)
            graph = np.zeros_like(corr, dtype=bool)
            seen = set()
            for i in np.argsort(-corr[a, b], kind="stable"):
                u, v = a[i], b[i]
                if u not in seen or v not in seen:
                    seen.update((u, v))
                    graph[u, v] = graph[v, u] = True
            np.testing.assert_array_equal(graph.sum(axis=1), saved["graph_degree"])
            info["shapg_components"] = connected_components(graph, directed=False, return_labels=False)
            assert len(x) == saved["construction_rows"]
        elif method == "banzhaf":
            assert (saved["complementary_pairs"], saved["sampled_masks"]) == (32, 64)
        elif method == "sampling_shap":
            assert saved["implementation"] == "shap.SamplingExplainer"
            assert len(saved["ranking_rows"]) <= 32 and len(saved["background_rows"]) <= 128
            assert saved["nsamples_per_explanation"] == len(cols) * 12
        elif method == "tree_shap_top_k":
            assert saved["ranking"] == "mean absolute exact raw-margin TreeSHAP across rows and outputs, excluding bias"
            assert saved["max_explanation_rows"] == 1024
    elif method == "minshap":
        assert saved["paths"] == 4 and saved["alpha"] == .05
        keep = ((np.asarray(saved["negative_certificate_path"]) < 0)
                & (np.asarray(saved["observed_minimum"]) >= np.asarray(saved["threshold_of_observed_minimum"])))
        selected = set(cols[keep])
    elif method == "nsga2":
        assert [saved[k] for k in ("population", "generations", "logical_evaluations")] == [8, 2, 24]
        winner = min(saved["pareto_front"], key=lambda s:(s["validation_utility_loss"],s["cardinality_ratio"],s["indices"]))
        assert winner == saved["chosen_point"]
        selected = set(cols[winner["indices"]])
    elif method == "gain":
        assert saved["aggregate"] == "gain"
        selected = set(saved["order"][:saved["k"]])
    elif method == "pfi":
        assert saved["repeats"] == 3 and len(saved["ranking_rows"]) <= 128
        score = np.array(saved["mean_drops"])
        selected = set(cols[score > 0]) or {cols[score.argmax()]}
    elif method == "full":
        selected = set(cols)
    else:
        raise KeyError(method)
    assert result["selected"] == [j for j in cols if j in selected]
    return info


def main():
    protocol = read(RUN / "protocol.json")
    for path, digest in protocol["source"].items():
        if "/baselines/" in path or path.endswith("literature_worker.py"):
            assert sha(ROOT / path) == digest
    assert sha(RUN / "original-protocol.json") == protocol["retention"]["original_protocol_sha256"]
    rows = []
    for name in protocol["datasets"]:
        for seed in protocol["seeds"]:
            folder = RUN / "inputs" / f"seed-{seed}" / name
            meta = read(folder / "metadata.json")
            y = np.load(folder / "test_y.npy")
            x = np.load(folder / "train.npy", mmap_mode="r")
            for file in ["train.npy", "train_y.npy", "test.npy", "test_y.npy"]:
                assert sha(folder / file) == meta["sha256"][file]
            for method in protocol["methods"] + protocol["controls"]:
                mode = "native" if method in protocol["methods"] else "shared"
                arm = RUN / "runs" / f"seed-{seed}" / name / mode / method
                result, saved = read(arm / "result.json"), read(arm / "estimator.json")
                frozen, gate = read(arm / "frozen.json"), read(arm / "test_access.json")
                assert result["status"] == "completed" and frozen["candidate"] == frozen["selected"] == result["selected"]
                assert not frozen["fallback"] and not frozen["enforce_guard"]
                signed = json.dumps(frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                assert hashlib.sha256(signed).hexdigest() == result["frozen_hash"] == gate["frozen_hash"]
                assert gate["test_y_sha256"] == meta["sha256"]["test_y.npy"] and gate["test_x_sha256"] == meta["sha256"]["test.npy"]
                extra = check_support(method, meta, saved, x, result)
                with np.load(arm / "predictions.npz") as pred:
                    np.testing.assert_array_equal(pred["y_test"], y)
                    p = pred["test_pred"]
                    if name == "abalone":
                        calculated = dict(mae=mean_absolute_error(y,p), r2=r2_score(y,p))
                    else:
                        calculated = dict(auc=auc(y,p), accuracy=accuracy_score(y,p.argmax(axis=1)))
                        if name == "credit_card_fraud":
                            calculated["ap"] = average_precision_score(y,p[:,1])
                    for metric, value in calculated.items():
                        # Historical evaluator stores a lower-clipped R²; the public seed
                        # table also includes the ordinary unclipped held-out coefficient.
                        protocol_score = max(0., value) if metric == "r2" else value
                        np.testing.assert_allclose(protocol_score, result["test_scores"][metric], rtol=0, atol=1e-10)
                rows.append(dict(dataset=name,seed=seed,method=method,k=result["k"],fits=result["fits"],**extra,**calculated))
    assert len(rows) == 270
    output = pd.DataFrame(rows)
    paired = output.merge(pd.read_csv(RUN/"per-seed.csv"), on=["dataset", "seed", "method"],
                          suffixes=("", "_published"), validate="one_to_one")
    assert len(paired) == len(rows)
    for metric in ("k", "mae", "r2", "auc", "accuracy", "ap"):
        np.testing.assert_allclose(paired[metric], paired[metric+"_published"], rtol=0, atol=1e-10)
    output.to_csv(HERE / "historical-controls.csv", index=False)
    print(f"Verified all {len(rows)} historical records")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
