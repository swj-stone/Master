"""Compare native RFE and univariate ANOVA at the widths of the separating selector."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import DATA, OUT as PRIMARY, ROOT, config, digest
from minppi_mofs.utility import compute_utility


OUT = ROOT / "results/context_study/matched-selectors"
DESIGN = ROOT / "configs/context_study/matched_selectors.json"


def run_fold(name, fold):
    source = PRIMARY / name / f"fold-{fold}"
    folder = OUT / name / f"fold-{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    donor = json.loads((source/"selection.json").read_bytes())
    receipt = json.loads((source/"completed.json").read_bytes())
    table = DATA / name / f"{name}.tsv.gz"
    assert digest(table) == receipt["data_sha256"] and digest(source/"selection.json") == receipt["selection_sha256"]
    frame = pd.read_csv(table, sep="\t")
    y = frame.pop("target").map({v:i for i,v in enumerate(donor["class_order"])}).to_numpy()
    x = frame.to_numpy(dtype=np.float32)
    dev = np.sort(np.r_[donor["train"], donor["rank"]])
    test = np.asarray(donor["test"])
    k = len(donor["selected"]["antichain"])
    seed = 20421 + fold
    start = perf_counter()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    xd = imputer.fit_transform(x[dev])
    xe = imputer.transform(x[test])
    prep_s = perf_counter()-start
    selector_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.1, tree_method="hist",
                                   device="cpu", n_jobs=1, importance_type="gain", random_state=seed)
    methods = {"rfe_same_k": RFE(selector_model,n_features_to_select=k,step=.2),
               "anova_same_k": SelectKBest(f_classif,k=k)}
    supports, rows, predictions = {}, {}, {}
    for arm, selector in methods.items():
        start = perf_counter()
        selector.fit(xd, y[dev])
        cols = np.flatnonzero(selector.get_support())
        assert len(cols) == k
        supports[arm] = cols.tolist()
        rows[arm] = dict(dataset=name, fold=fold, arm=arm, k=k, width=x.shape[1],
                         prep_s=prep_s, selection_s=perf_counter()-start)
    store(folder/"selection.json", dict(selected=supports, donor=receipt["selection_sha256"],
                                       development=dev.tolist(), test=test.tolist()))
    for arm, cols in supports.items():
        start = perf_counter()
        p = fit(xd[:,cols], y[dev], seed).predict_proba(xe[:,cols])
        rows[arm].update(refit_s=perf_counter()-start, auc=compute_utility(y[test],p,metric="ovr_auc"),
                         accuracy=float((p.argmax(axis=1)==y[test]).mean()))
        predictions[arm] = p
    np.savez_compressed(folder/"predictions.npz", row_id=test, y=y[test], **predictions)
    store(folder/"completed.json", dict(rows=list(rows.values()), protocol=digest(OUT/"protocol.json"),
          supports=digest(folder/"selection.json"), probabilities=digest(folder/"predictions.npz")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2)
    args = parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            return run_fold(args.worker[0], int(args.worker[1]))
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT/"protocol.json"
    lock = {p.name:digest(p) for p in [DESIGN, Path(__file__), PRIMARY/"protocol-lock.json",
                                       ROOT/"experiments/context_study/benchmark.py"]}
    if protocol.exists() and json.loads(protocol.read_bytes()) != lock:
        raise ValueError("A frozen input or protocol changed")
    store(protocol,lock)
    rows = []
    for name in config()["datasets"]:
        for fold in range(3):
            folder = OUT/name/f"fold-{fold}"
            if not (folder/"completed.json").exists():
                subprocess.run([sys.executable,"-m","experiments.context_study.matched_selectors",
                                "--worker",name,str(fold)], timeout=3600,check=True)
            rec = json.loads((folder/"completed.json").read_bytes())
            assert rec["protocol"] == digest(protocol) and rec["supports"] == digest(folder/"selection.json")
            assert rec["probabilities"] == digest(folder/"predictions.npz")
            rows.extend(rec["rows"])
        print(f"Completed {name}: {len(rows)}/84 matched-width baselines",flush=True)
    if len(rows) != 84:
        raise ValueError("Incomplete comparison")
    pd.DataFrame(rows).to_csv(OUT/"per-fold.csv",index=False)


if __name__ == "__main__":
    main()
