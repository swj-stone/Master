"""Independent nested RFECV control on the same outer folds as the primary PMLB study."""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import DATA, OUT, ROOT, config, digest
from minppi_mofs.utility import compute_utility

STUDY = OUT.parent / "prospective-pmlb-rfecv-r1"
DESIGN = ROOT / "configs/context_study/rfecv.json"


def run_fold(name, fold):
    base = OUT / name / f"fold-{fold}"
    folder = STUDY / name / f"fold-{fold}"
    folder.mkdir(exist_ok=True, parents=True)
    selection = json.loads((base/"selection.json").read_bytes())
    receipt = json.loads((base/"completed.json").read_bytes())
    if digest(base/"selection.json") != receipt["selection_sha256"]:
        raise ValueError("Primary split receipt differs")
    table = DATA / name / f"{name}.tsv.gz"
    if digest(table) != receipt["data_sha256"]:
        raise ValueError("Source differs from primary experiment")
    frame = pd.read_csv(table,sep="\t")
    classes = {v:i for i,v in enumerate(selection["class_order"])}
    y = frame.pop("target").map(classes).to_numpy(dtype=int)
    x = frame.to_numpy(dtype=np.float32)
    test = np.asarray(selection["test"])
    dev = np.sort(np.r_[selection["train"],selection["rank"]])
    seed = 20421+fold
    pipeline = Pipeline([("imputer",SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("classifier",XGBClassifier(n_estimators=100,max_depth=4,learning_rate=.1,
                           tree_method="hist",device="cpu",n_jobs=1,importance_type="gain",random_state=seed))])
    score = "roc_auc" if len(classes)==2 else "roc_auc_ovr"
    inner = StratifiedKFold(3,shuffle=True,random_state=31173+fold)
    selector = RFECV(estimator=pipeline,step=.2,min_features_to_select=1,cv=inner,
                     scoring=score,importance_getter="named_steps.classifier.feature_importances_",n_jobs=1)
    start = perf_counter()
    selector.fit(x[dev],y[dev])
    elapsed = perf_counter()-start
    selected = np.flatnonzero(selector.support_).tolist()
    paths = {k:v.tolist() for k,v in selector.cv_results_.items()
             if k in {"mean_test_score","std_test_score","n_features"} or k.endswith("_test_score")}
    store(folder/"selection.json", dict(primary_selection_sha256=receipt["selection_sha256"], selected=selected,
          test=test.tolist(), inner=[(dev[a].tolist(),dev[b].tolist()) for a,b in inner.split(dev,y[dev])],
          scoring=score, cv_results=paths))
    start = perf_counter()
    imputer = SimpleImputer(strategy="median",keep_empty_features=True)
    train = imputer.fit_transform(x[dev][:,selected])
    eval_x = imputer.transform(x[test][:,selected])
    pred = fit(train,y[dev],seed).predict_proba(eval_x)
    final_s = perf_counter()-start
    auc = compute_utility(y[test],pred,metric="ovr_auc")
    acc = float(accuracy_score(y[test],pred.argmax(axis=1)))
    np.savez_compressed(folder/"predictions.npz",row_id=test,y=y[test],rfecv=pred)
    row = dict(dataset=name,fold=fold,arm="rfecv",n=len(y),width=x.shape[1],k=len(selected),
               n_test=len(test),auc=auc,accuracy=acc,selection_s=elapsed,final_s=final_s,total_s=elapsed+final_s,
               selected=json.dumps(selected))
    store(folder/"completed.json",dict(protocol_sha256=digest(STUDY/"protocol-lock.json"),record=row,
          selection_sha256=digest(folder/"selection.json"),predictions_sha256=digest(folder/"predictions.npz")))


def main():
    STUDY.mkdir(parents=True,exist_ok=True)
    primary = OUT/"protocol-lock.json"
    lock = dict(protocol=json.loads(DESIGN.read_text(encoding="utf-8-sig")),source_sha256=digest(Path(__file__)),
                specification_sha256=digest(DESIGN),primary_sha256=digest(primary))
    lockfile = STUDY/"protocol-lock.json"
    if lockfile.exists() and json.loads(lockfile.read_bytes()) != lock:
        raise RuntimeError("RFECV protocol changed")
    store(lockfile,lock)
    records,errors = [],[]
    def log(message):
        entry = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
        with (STUDY/"run.log").open("a",encoding="utf8") as stream:
            print(entry,file=stream)
        print(entry,flush=True)
    for name in config()["datasets"]:
        for fold in range(3):
            folder = STUDY/name/f"fold-{fold}"
            done = folder/"completed.json"
            if not done.exists():
                log(f"Starting {name} fold {fold+1}/3")
                start = perf_counter()
                try:
                    env = os.environ.copy()
                    env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
                    subprocess.run([sys.executable,"-m","experiments.context_study.rfecv","--worker",name,str(fold)],
                                   cwd=ROOT,timeout=3600,check=True,env=env)
                except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
                    errors.append((name,fold,str(exc)))
                    log(f"FAILED {name} fold {fold}: {exc}")
                    continue
                log(f"Completed {name} fold {fold+1} in {perf_counter()-start:.1f}s")
            result = json.loads(done.read_bytes())
            if (digest(lockfile) != result["protocol_sha256"]
                    or digest(folder/"predictions.npz") != result["predictions_sha256"]
                    or digest(folder/"selection.json") != result["selection_sha256"]):
                raise ValueError(f"Bad receipt: {folder}")
            records.append(result["record"])
    if records:
        with (STUDY/"per-fold.csv").open("w",encoding="utf8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    store(STUDY/"failures.json",errors)
    if errors:
        raise RuntimeError(f"{len(errors)} failed folds")
    log(f"Completed all {len(records)} held-out comparisons")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker",nargs=2,metavar=("DATASET","FOLD"))
    args=p.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            run_fold(args.worker[0],int(args.worker[1]))
    else:
        main()

