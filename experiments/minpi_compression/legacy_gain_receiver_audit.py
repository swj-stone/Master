"""Audit every strong/fixed selection against its saved inner curves and test scores."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.minpi_compression import benchmark as bench
from experiments.minpi_compression import legacy_gain as legacy
from experiments.minpi_compression.legacy_gain_receiver import as_score_matrix


def check(history_file=None):
    dest = legacy.DEST/"xgb-receiver"
    table = pd.read_csv(dest/"scores.csv").set_index(["dataset","seed","method","part"])
    archive_metrics = pd.read_csv(bench.ROOT/"docs/thesis/archives/piv1/final_results_by_seed.csv").set_index(
        ["dataset","seed"])
    counts = dict(frozen_refits=0, fresh_guards=0, historical_scores=0)
    if history_file is None:
        history_file = bench.ROOT/"packages/thesis-piv1-evidence.zip"
    with ZipFile(dest/"forecasts.zip") as nonlinear, ZipFile(legacy.DEST/"forecasts.zip") as original, ZipFile(history_file) as historical:
        lock = json.loads(nonlinear.read("lock.json"))
        assert len(lock) == 30 and nonlinear.testzip() is None and original.testzip() is None
        largest_archive_difference = 0.0
        for key, expected in lock.items():
            encoded = nonlinear.read(key+"/selection.json")
            assert sha256(encoded).hexdigest() == expected
            meta = json.loads(encoded)
            target = np.load(BytesIO(original.read(key+"/targets.npy")),allow_pickle=False)
            original_bytes = original.read(key+"/selection.json")
            assert meta["parent"] == sha256(original_bytes).hexdigest()
            parent = json.loads(original_bytes)
            records = {r["method"]:r for r in parent["rows"] if r["repeat"]==0}
            n,s,task = parent["dataset"],parent["seed"],parent["task"]
            pred = nonlinear.read(key+"/predictions.npz")
            assert sha256(pred).hexdigest() == meta["forecast"]
            with np.load(BytesIO(pred)) as forecasts:
                for arm, record in meta["records"].items():
                    check_type, method = arm.split("_")
                    old = records[method]
                    rank, p = np.array(old["order"]), old["p"]
                    if check_type == "guard" and method != "full":
                        trace = {int(k):v for k,v in record["trace"].items()}
                        assert p in trace and set(trace) == set(map(int,old["inner"]))
                        expected_k = min(k for k,v in trace.items() if bench.eligible(v,trace[p],task))
                    else:
                        expected_k = old["k"]
                    assert expected_k == record["k"]
                    np.testing.assert_array_equal(record["selected"],np.sort(rank[:expected_k]))
                    metrics = bench.metrics(target,forecasts[arm],task,parent["scale"])
                    row = table.loc[(n,s,arm,"primary")]
                    for m,v in metrics.items():
                        np.testing.assert_allclose(row[m],v,atol=1e-12,rtol=0)
                    counts["fresh_guards" if check_type=="guard" else "frozen_refits"] += 1
            old_summary = archive_metrics.loc[n,s]
            for arm,prefix in [("old_direct_gain","gain"),("old_PIv1","piv1")]:
                row = table.loc[(n,s,arm,"primary")]
                suffix = f"/seed-{s}/{n}/{arm.removeprefix('old_')}/predictions.npz"
                member, = [f for f in historical.namelist() if "PIv1-all-seeds-" in f and f.endswith(suffix)]
                with np.load(BytesIO(historical.read(member))) as payload:
                    np.testing.assert_array_equal(target,payload["y"])
                    prediction = as_score_matrix(payload["selected"],task)
                common = bench.metrics(target,prediction,task,parent["scale"])
                for m in (("auc","accuracy") if task=="classification" else ("r2","mae")):
                    # The legacy binary AUC uses only the positive score. The common receiver
                    # score averages both OvR AUCs; float32 complements can introduce ties.
                    native = roc_auc_score(target,prediction[:,1]) if m=="auc" and n!="isolet" else common[m]
                    np.testing.assert_allclose(row[m],common[m],atol=1e-12,rtol=0)
                    difference = abs(row[m] - old_summary[f"{prefix}_test_{m}"])
                    largest_archive_difference = max(largest_archive_difference,difference)
                    np.testing.assert_allclose(native,old_summary[f"{prefix}_test_{m}"],atol=1e-7,rtol=0)
                counts["historical_scores"] += 1
    assert counts == dict(frozen_refits=120, fresh_guards=120, historical_scores=60)
    print(counts, "maximum discrepancy against metrics in the historical table",largest_archive_difference)


if __name__=="__main__":
    check()
