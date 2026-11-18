"""Crossed four-stage feature study. Run in history with thesis-python.ps1."""
import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import sklearn
from sklearn.datasets import load_wine
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.stable_minpi_stress import inputs, ROOT, digest
from experiments.signal_vote_real_proxies import SOURCES
from experiments.compare_predictive_minpi import choose
from research.algorithms.coalition_residual import select_features
from research.four_stage.pfi import pfi_top

DATASETS = ("digits","wine","breast_cancer","spambase","banknote","madelon",
            "diabetes","abalone","airfoil","crimedata")
SEEDS = (13591,34841,64109)
ARMS = ("full","minpi80","F80","random80","PFI80","minpi_auto","F_match","random_match","PFI_match")
MODELS = ("ridge","ridge_ple","mlp","mlp_ple")


def load(name):
    if name in SOURCES:
        member, delimiter, task, sha = SOURCES[name]
        path = ROOT / f"data/raw/uci-{name}.zip"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        with ZipFile(path) as z, z.open(member) as f:
            a = np.loadtxt(f, delimiter=delimiter)
        X, y = a[:,:-1], a[:,-1]
    elif name == "wine":
        X, y = load_wine(return_X_y=True)
        task = "classification"
    else:
        X, y, task, groups, _, dup = inputs(name)
        return X, y, task, groups
    _, ids = np.unique(X, axis=0, return_inverse=True)
    return X, y, task, ids if len(set(ids)) < len(X) else None


def encode(A, B, ple):
    # Numeric imputation and every knot are fit on the outer training set.
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        valid = np.isfinite(A)
        mean = np.where(valid,A,0.).sum(axis=0) / np.maximum(valid.sum(axis=0),1)
        A, B = (np.where(np.isfinite(x),x,mean) for x in (A,B))
    if ple:
        cuts = np.quantile(A,[0., .25, .50, .75, 1.], axis=0).T
        spans = np.diff(cuts,axis=1)
        # Zero-length intervals contribute no coordinate.
        def transform(x):
            z = np.clip((x[:,:,None]-cuts[:,:-1])/np.where(spans>0, spans, 1.),0.,1.)
            return z[:,spans>0]
        A,B = transform(A),transform(B)
        if A.shape[1] == 0:
            A,B = np.zeros((len(A),1)),np.zeros((len(B),1))
    scale = StandardScaler().fit(A)
    return scale.transform(A).astype(np.float32), scale.transform(B).astype(np.float32)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    protocol = dict(datasets=DATASETS,seeds=SEEDS,arms=ARMS,models=MODELS,
        numpy=np.__version__,sklearn=sklearn.__version__,mode="one-shot breadth screen",
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (
        Path(__file__).resolve(), ROOT/"research/four_stage/pfi.py", ROOT/"research/algorithms/coalition_residual.py",ROOT/"experiments/compare_predictive_minpi.py",ROOT/"experiments/stable_minpi_stress.py")})
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2))
    with (output/"outcomes.csv").open("x",newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(("source","seed","model","arm","p","k","score","select_ms","total_ms","converged"))
        for name in DATASETS:
            X, y, task, groups = load(name)
            classifier = task == "classification"
            for seed in SEEDS:
                generator = ((StratifiedGroupKFold if classifier else GroupKFold) if groups is not None
                             else (StratifiedKFold if classifier else KFold))
                tr,te = next(generator(3,shuffle=True,random_state=seed).split(X,y,groups))
                if groups is not None:
                    assert not np.intersect1d(groups[tr],groups[te]).size
                a,b = X[tr],X[te]
                labels = y[tr]
                p = a.shape[1]
                width = len(select_features(a,labels,task=task,random_state=seed)[0])
                arrays = dict(train=tr,test=te,truth=y[te],input_X_sha256=np.array(digest(X)),input_y_sha256=np.array(digest(y)))
                order = np.random.default_rng(seed).permutation(ARMS)
                for model in MODELS:
                    for arm in order:
                        k = width if arm.endswith(('auto','match')) else (4*p+4)//5
                        start = perf_counter()
                        if arm=="full":
                            cols=np.arange(p)
                        elif arm.startswith("minpi"):
                            cols,_=select_features(a,labels,count=None if arm=="minpi_auto" else k,task=task,random_state=seed)
                        elif arm.startswith("random"):
                            cols=choose("random",a,labels,task,seed,k)
                        elif arm.startswith("PFI"):
                            cols=pfi_top(a,labels,task,seed,k)
                        elif arm.startswith("F"):
                            A=a
                            if not np.isfinite(A).all():
                                valid=np.isfinite(A)
                                A=np.where(valid,A,np.where(valid,A,0.).sum(axis=0)/np.maximum(1,valid.sum(axis=0)))
                            # Center in float64 before the library's raw-sum ANOVA.
                            # This leaves the theoretical F unchanged for offset pixels/sensors.
                            A = np.asarray(A,dtype=np.float64)
                            cols=choose("f",A-A.mean(axis=0),labels,task,seed,k)
                        else:
                            raise ValueError(arm)
                        selected = perf_counter()
                        A,B=(a,b) if len(cols)==p else (a[:,cols],b[:,cols])
                        A,B=encode(A,B,model.endswith("_ple"))
                        fit_y=labels
                        if model.startswith("ridge"):
                            estimator=(RidgeClassifier if classifier else Ridge)(alpha=1.,solver="cholesky")
                        else:
                            estimator=(MLPClassifier if classifier else MLPRegressor)(
                                hidden_layer_sizes=(32,),max_iter=40,random_state=seed,
                                learning_rate_init=.003,batch_size=min(128,len(A)))
                            if not classifier:
                                fit_y=(labels-labels.mean())/(labels.std() or 1.)
                        with warnings.catch_warnings(record=True) as recorded:
                            warnings.simplefilter("always",ConvergenceWarning)
                            estimator.fit(A,fit_y)
                        pred=estimator.predict(B)
                        if not classifier and model.startswith("mlp"):
                            pred=pred*(labels.std() or 1.)+labels.mean()
                        end=perf_counter()
                        score=(balanced_accuracy_score if classifier else r2_score)(y[te],pred)
                        if arm in arrays:
                            np.testing.assert_array_equal(cols,arrays[arm])
                        arrays[arm]=cols
                        arrays[model+"_"+arm+"_pred"]=pred
                        writer.writerow((name,seed,model,arm,p,len(cols),score,(selected-start)*1e3,
                                         (end-start)*1e3,not any(issubclass(w.category,ConvergenceWarning) for w in recorded)))
                np.savez_compressed(output/f"{name}-{seed}.npz",**arrays)
                fp.flush()
            print("completed",name,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    with threadpool_limits(1):
        run(parser.parse_args().output)
