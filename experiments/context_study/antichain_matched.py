"""Refit the same-width Gain prefix donated by each frozen antichain selection."""
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
from experiments.context_study.data import CONFIG, DATA, OUT, ROOT,config,digest
from minppi_mofs.utility import compute_utility

DESIGN=ROOT/"configs/context_study/antichain-matched.json"
STUDY=OUT.parent/"prospective-pmlb-antichain-matched-r1"
ARM="antichain_matched_gain"


def run_fold(name,fold):
    parent=OUT/name/f"fold-{fold}"
    done=json.loads((parent/"completed.json").read_bytes())
    sel=json.loads((parent/"selection.json").read_bytes())
    table=DATA/name/(name+".tsv.gz")
    if digest(parent/"selection.json")!=done["selection_sha256"] or digest(table)!=done["data_sha256"]:
        raise ValueError("Reference identity differs")
    donor=sel["selected"]["antichain"]
    selected=sorted(sel["order"][:len(donor)])
    dev=np.sort(np.r_[sel["train"],sel["rank"]])
    test=np.asarray(sel["test"])
    folder=STUDY/name/f"fold-{fold}"
    folder.mkdir(parents=True,exist_ok=True)
    store(folder/"selection.json",dict(selected=selected,donor=donor,test=test.tolist(),
          donor_selection_sha256=done["selection_sha256"],note="Width supplied by primary antichain arm"))
    frame=pd.read_csv(table,sep="\t")
    classes={v:i for i,v in enumerate(sel["class_order"])}
    y=frame.pop("target").map(classes).to_numpy(dtype=int)
    x=frame.to_numpy(dtype=np.float32)
    start=perf_counter()
    imputer=SimpleImputer(strategy="median",keep_empty_features=True)
    xd,xe=imputer.fit_transform(x[dev]),imputer.transform(x[test])
    model=fit(xd[:,selected],y[dev],20421+fold)
    pred=model.predict_proba(xe[:,selected])
    elapsed=perf_counter()-start
    row=dict(dataset=name,fold=fold,arm=ARM,k=len(selected),width=x.shape[1],n_test=len(test),
             auc=compute_utility(y[test],pred,metric="ovr_auc"),accuracy=float((pred.argmax(axis=1)==y[test]).mean()),
             conditional_refit_s=elapsed,selected=json.dumps(selected))
    np.savez_compressed(folder/"predictions.npz",row_id=test,y=y[test],**{ARM:pred})
    store(folder/"completed.json",dict(record=row,selection_sha256=digest(folder/"selection.json"),
          predictions_sha256=digest(folder/"predictions.npz"),protocol_sha256=digest(STUDY/"protocol-lock.json")))


def main():
    STUDY.mkdir(parents=True,exist_ok=True)
    lock=dict(primary_sha256=digest(OUT/"protocol-lock.json"),code_sha256=digest(Path(__file__)),
              design_sha256=digest(DESIGN),design=json.loads(DESIGN.read_text(encoding="utf-8-sig")))
    path=STUDY/"protocol-lock.json"
    if path.exists() and json.loads(path.read_bytes())!=lock:
        raise ValueError("Changed conditional benchmark protocol")
    store(path,lock)
    records,failures=[],[]
    for name in config()["datasets"]:
        for fold in range(3):
            folder=STUDY/name/f"fold-{fold}"
            if not (folder/"completed.json").exists():
                print(f"Refitting {name} fold {fold+1}/3",flush=True)
                try:
                    subprocess.run([sys.executable,"-m","experiments.context_study.antichain_matched","--worker",name,str(fold)],
                                   cwd=ROOT,timeout=3600,check=True)
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
                    failures.append((name,fold,str(exc)))
                    continue
            r=json.loads((folder/"completed.json").read_bytes())
            if (r["protocol_sha256"]!=digest(path) or r["selection_sha256"]!=digest(folder/"selection.json")
                  or r["predictions_sha256"]!=digest(folder/"predictions.npz")):
                raise ValueError(f"Invalid receipt at {folder}")
            records.append(r["record"])
    pd.DataFrame(records).to_csv(STUDY/"per-fold.csv",index=False)
    store(STUDY/"failures.json",failures)
    if failures:
        raise RuntimeError(f"{len(failures)} failed folds")
    print(f"Completed {len(records)} conditional same-width refits")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=2)
    args=parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            run_fold(args.worker[0],int(args.worker[1]))
    else:
        main()

