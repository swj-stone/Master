"""Audit compact-route prediction matrices and describe fixed-model paired test loss uncertainty."""
import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.test.piv3_lite_uncertainty import _csv, _paired_scores, _quantile, _relative_loss
from experiments.test.run_compact_minpi import BASE, DATASETS, DEFAULT, DEFAULT_ROOT, _sha, read
from minppi_mofs.experiment_protocol import canonical_hash

CHALLENGERS = ("tree_min", "pair_min", "tree_min_pfi", "pair_min_pfi")
REFERENCES = ("full", "pfi")
N_BOOT = 399


def _load(root, dataset, method, metadata, labels):
    folder = root/dataset/method
    timer = read(folder/"watchdog.json")
    if timer["status"] != "completed":
        return None
    record = read(folder/"result.json")
    frozen = read(folder/"frozen.json")
    event = read(folder/"test_access.json")
    if (record != timer["result"] or record["seed"] != 42 or record["dataset"] != dataset or
            record["method"] != method or record["selected"] != frozen["selected"] or
            record["frozen_hash"] != canonical_hash(frozen) or record["frozen_hash"] != event["frozen_hash"] or
            event["test_y_sha256"] != metadata["sha256"]["test_y.npy"] or
            event["test_x_sha256"] != metadata["sha256"]["test.npy"]):
        raise ValueError(f"Inconsistent run or test identity: {folder}")
    with np.load(folder/"predictions.npz", allow_pickle=False) as archive:
        y, pred = archive["y_test"], archive["test_pred"]
    if not np.array_equal(y, labels) or not np.isfinite(pred).all():
        raise ValueError(f"Prediction archive disagrees with frozen labels or contains nonfinite values: {folder}")
    expected = labels.shape if dataset == "abalone" else (len(labels),len(np.unique(labels)))
    if pred.shape != expected or (pred.ndim == 2 and
            (np.min(pred) < 0 or np.max(pred) > 1 or not np.allclose(pred.sum(axis=1),1,atol=1e-5))):
        raise ValueError(f"Invalid shape/probabilities: {folder}")
    return pred, record["test_scores"]


def report(run, reference_run=BASE, challengers=CHALLENGERS):
    run, reference_run = Path(run), Path(reference_run)
    protocol = read(run/"protocol.json")
    primary = read(reference_run/"protocol.json")
    if (protocol["benchmark_sha256"] != _sha(reference_run/"protocol.json") or protocol["data"] != primary["data"] or
            protocol["seed"] != primary["seed"] or primary["seed"] != 42):
        raise ValueError("Primary benchmark identity differs")
    metric_rows, joint_rows = [], []
    for i,dataset in enumerate(DATASETS):
        dataset_path = DEFAULT_ROOT/dataset
        if _sha(dataset_path/"metadata.json") != protocol["data"][dataset]["hash"]:
            raise ValueError(f"Changed input metadata: {dataset}")
        meta = read(dataset_path/"metadata.json")
        if _sha(dataset_path/"test_y.npy") != meta["sha256"]["test_y.npy"]:
            raise ValueError(f"Changed test labels: {dataset}")
        labels = np.load(dataset_path/"test_y.npy", allow_pickle=False)
        predictions, expected = {}, {}
        for arm in (*REFERENCES,*challengers):
            loaded = _load(reference_run if arm in REFERENCES else run,dataset,arm,meta,labels)
            if loaded is not None:
                predictions[arm], expected[arm] = loaded
        with threadpool_limits(limits=1):
            points, boots = _paired_scores(dataset,labels,predictions,N_BOOT,np.random.default_rng([42,i]))
        for arm, scores in points.items():
            if (set(scores) != set(expected[arm]) or any(abs(v-expected[arm][k])>1e-12 for k,v in scores.items())):
                raise ValueError(f"Test metrics disagree: {dataset}/{arm}")
        for arm in challengers:
            detail=dict(dataset=dataset,method=arm,n_test=len(labels),n_boot=N_BOOT,seed=42,
                        scope="paired percentile bootstrap, fixed fitted models and fixed class counts for classification")
            if any(m not in predictions for m in (arm,*REFERENCES)):
                joint_rows.append(dict(**detail,status="unavailable",loss_limit=.05))
                continue
            contrasts, point_loses = [], []
            for ref in REFERENCES:
                for k in points[arm]:
                    v, target = points[arm][k], points[ref][k]
                    loss = _relative_loss(v,target,k)
                    draws = np.fromiter((_relative_loss(x,y,k) for x,y in zip(boots[arm][k],boots[ref][k],strict=True)),dtype=float)
                    upper = _quantile(draws)
                    metric_rows.append(dict(**detail,reference=ref,metric=k,score=v,reference_score=target,
                                            relative_loss=loss,upper95_relative_loss=upper))
                    point_loses.append(loss)
                    contrasts.append(draws)
            worst = max(point_loses)
            upper = _quantile(np.max(contrasts,axis=0))
            joint_rows.append(dict(**detail,status="completed",worst_point_relative_loss=worst,
                                   upper95_worst_relative_loss=upper,loss_limit=.05,point_within=worst<=.05+1e-12,
                                   upper_within=upper<=.05+1e-12))
    if metric_rows:
        _csv(run/"quality-by-metric.csv",metric_rows)
    _csv(run/"quality-bounds.csv",joint_rows)
    return joint_rows


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    parser.add_argument("--reference-run",type=Path,default=BASE)
    parser.add_argument("--methods",nargs="+",default=CHALLENGERS)
    args=parser.parse_args()
    rows=report(args.run,args.reference_run,args.methods)
    print(f"{sum(r.get('point_within',False) for r in rows)}/{len(rows)} point comparisons within 5%; "
          f"{sum(r.get('upper_within',False) for r in rows)}/{len(rows)} bootstrap upper estimates within 5%.")

