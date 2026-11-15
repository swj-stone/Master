"""Historical direct Gain width-selection rule on the archived CPU benchmark splits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import ROOT, DATA, OUT as PRIMARY, config, digest
from minppi_mofs.utility import compute_utility

OUT = ROOT/"results/context_review/exhaustive-gain"
DESIGN = ROOT/"configs/context_review/exhaustive_gain.json"


def run(name, fold):
    parent = PRIMARY/name/f"fold-{fold}"
    folder = OUT/name/f"fold-{fold}"
    folder.mkdir(parents=True,exist_ok=True)
    split = json.loads((parent/"selection.json").read_bytes())
    receipt = json.loads((parent/"completed.json").read_bytes())
    table = DATA/name/f"{name}.tsv.gz"
    assert digest(table)==receipt["data_sha256"] and digest(parent/"selection.json")==receipt["selection_sha256"]
    frame = pd.read_csv(table,sep="\t")
    y = frame.pop("target").map({v:i for i,v in enumerate(split["class_order"])}).to_numpy()
    x = frame.to_numpy(dtype=np.float32)
    train, rank, test = [np.asarray(split[k]) for k in ("train","rank","test")]
    seed = 20421+fold
    start = perf_counter()
    imp = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt,xv = imp.transform(x[train]),imp.transform(x[rank])
    model = fit(xt,y[train],seed)
    gain = model.get_booster().get_score(importance_type="gain")
    order = sorted(range(x.shape[1]),key=lambda i:(gain.get(f"f{i}",0.),i),reverse=True)
    full = model.predict_proba(xv)
    curve, best, chosen = [], -float("inf"), []
    for k in range(1,len(order)+1):
        cols = sorted(order[:k])
        p = full if k==len(order) else fit(xt[:,cols],y[train],seed).predict_proba(xv[:,cols])
        auc = float(compute_utility(y[rank],p,metric="ovr_auc"))
        if not np.isfinite(auc):
            raise ValueError("Non-finite validation AUC")
        curve.append(auc)
        if auc>best+1e-14:
            best,chosen = auc,cols
    selection_s = perf_counter()-start
    store(folder/"selection.json",dict(order=order,rank_auc=curve,chosen=chosen,source=receipt["selection_sha256"]))
    start = perf_counter()
    dev = np.sort(np.r_[train,rank])
    final = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[dev])
    xd,xe = final.transform(x[dev]),final.transform(x[test])
    p = fit(xd[:,chosen],y[dev],seed).predict_proba(xe[:,chosen])
    refit_s = perf_counter()-start
    np.savez_compressed(folder/"predictions.npz",y=y[test],row_id=test,probability=p)
    row = dict(dataset=name,fold=fold,width=len(order),k=len(chosen),fits=len(order)+1,
               auc=compute_utility(y[test],p,metric="ovr_auc"),accuracy=float((p.argmax(axis=1)==y[test]).mean()),
               selection_s=selection_s,refit_s=refit_s,total_s=selection_s+refit_s)
    store(folder/"completed.json",dict(row=row,selection=digest(folder/"selection.json"),
          predictions=digest(folder/"predictions.npz"),protocol=digest(OUT/"protocol.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=2)
    args=parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            return run(args.worker[0],int(args.worker[1]))
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[DESIGN,Path(__file__),PRIMARY/"protocol-lock.json",ROOT/"experiments/test/run_piv0_seed42.py"]
    lock={p.name:digest(p) for p in sources}
    path=OUT/"protocol.json"
    if path.exists() and json.loads(path.read_bytes())!=lock:
        raise ValueError("Changed locked design or code")
    store(path,lock)
    rows=[]
    for name in config()["datasets"]:
        for fold in range(3):
            folder=OUT/name/f"fold-{fold}"
            if not (folder/"completed.json").exists():
                subprocess.run([sys.executable,"-m","experiments.context_review.exhaustive_gain","--worker",
                                name,str(fold)],timeout=3600,check=True)
            rec=json.loads((folder/"completed.json").read_bytes())
            assert rec["protocol"]==digest(path) and rec["selection"]==digest(folder/"selection.json")
            assert rec["predictions"]==digest(folder/"predictions.npz")
            rows.append(rec["row"])
        print(f"{name}: {len(rows)}/42 folds",flush=True)
    pd.DataFrame(rows).to_csv(OUT/"per-fold.csv",index=False)


if __name__=="__main__":
    main()
