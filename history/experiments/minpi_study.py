"""Fixed-budget minPI study. Run through thesis-python.ps1 -m experiments.minpi_study."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import warnings
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, make_classification
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import f_classif, f_regression
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from minppi_mofs.exact_minpi import ExactMinPI, pivotal_rates, witness_margins
from minppi_mofs.piv3_compact_filter import _diverse_order, _votes

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/minpi-study.json'
P = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
OUTPUT = ROOT / 'results/minpi-study'
RAW = {'madelon', 'gisette', 'spambase', 'crimedata', 'isolet'}
SOURCES = [CONFIG, Path(__file__), ROOT/'src/minppi_mofs/exact_minpi.py',
           ROOT/'src/minppi_mofs/piv2.py', ROOT/'src/minppi_mofs/piv3_compact_filter.py']


def file_hash(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def load_dataset(name):
    task = 'regression' if name in {'diabetes', 'crimedata'} else 'classification'
    truth = []
    if name in RAW:
        meta = yaml.safe_load((ROOT/f'configs/{name}.yaml').read_bytes())['dataset']
        path = ROOT / meta['raw_path']
        source_hash = file_hash(path)
        if source_hash != meta['raw_sha256']:
            raise ValueError(f'{name} source digest mismatch')
        frame = pd.read_csv(path)
        y = frame.pop(meta['target']).to_numpy()
        if name == 'crimedata':
            frame = frame.loc[:, 'population':'LemasPctOfficDrugUn']
        names = list(map(str, frame.columns))
        X = frame.to_numpy(dtype=np.float32)
    elif name in {'digits', 'breast_cancer', 'diabetes'}:
        data = {'digits': load_digits, 'breast_cancer': load_breast_cancer,
                'diabetes': lambda: load_diabetes(scaled=False)}[name]()
        X, y = data.data.astype(np.float32), data.target
        names = list(map(str, data.feature_names))
        source_hash = f'sklearn-{sklearn.__version__}'
    else:
        rng = np.random.default_rng(P['data_simulation_seed'])
        X = rng.normal(size=(1200,128)).astype(np.float32)
        if name == 'sparse_signal':
            X, y = make_classification(n_samples=1200, n_features=128, n_informative=8,
                                       n_redundant=0, n_repeated=0, n_clusters_per_class=1,
                                       class_sep=1.5, flip_y=.02, shuffle=False,
                                       random_state=P['data_simulation_seed'])
            X = X.astype(np.float32)
            truth = list(range(8))
        elif name == 'xor':
            y = (X[:, 0] > 0) ^ (X[:, 1] > 0)
            truth = [0, 1]
        elif name == 'null':
            y = rng.integers(0, 2, len(X))
        else:
            raise KeyError(name)
        names = [f'x{i}' for i in range(X.shape[1])]
        source_hash = f"simulation-{P['data_simulation_seed']}-sklearn-{sklearn.__version__}"
    y = np.asarray(y)
    if not np.isfinite(X).all() or pd.isna(y).any():
        raise ValueError(f'{name}: unexpected missing/nonfinite input')
    if task == 'classification':
        y = LabelEncoder().fit_transform(y)
    order = np.random.default_rng(P['column_permutation']).permutation(X.shape[1])
    return np.ascontiguousarray(X[:, order]), y, task, [names[j] for j in order], order, truth, source_hash


def forest(task, seed):
    estimator = ExtraTreesClassifier if task == 'classification' else ExtraTreesRegressor
    return estimator(n_estimators=P['forest_trees'], max_features='sqrt', random_state=seed, n_jobs=1)


def mc_rates(votes, tiers, seed, draws=32):
    rng = np.random.default_rng(seed)
    n, p = votes.shape
    rates = []
    for k in tiers:
        if k == 0:
            rates.append(votes.mean(0))
            continue
        accumulated = np.zeros(p, dtype=np.float64)
        for _ in range(draws):
            order = rng.permutation(p)
            inside = order[:k]
            total = votes[:, inside].sum(axis=1, dtype=np.int16)
            delta = np.sign(total[:, None] + votes) - np.sign(total[:, None])
            without = total[:, None] - votes[:, inside] + votes[:, order[k], None]
            delta[:, inside] = np.sign(without + votes[:, inside]) - np.sign(without)
            accumulated += delta.sum(axis=0)
        rates.append(accumulated / (n * draws))
    return np.asarray(rates)


def select(method, X, y, task, seed):
    p = X.shape[1]
    k = ceil(P['selection_fraction'] * p)
    diagnostics = {}
    if method == 'full':
        return np.arange(p), diagnostics
    if method == 'random':
        ranking = np.random.default_rng(seed).permutation(p)
    elif method == 'f':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore', category=UserWarning)
            score = (f_classif if task == 'classification' else f_regression)(X, y)[0]
        ranking = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind='stable')
    elif method in {'forest', 'exact_forest'}:
        cols = np.arange(p)
        if method == 'exact_forest':
            screen = ExactMinPI(task=task, random_state=seed, fraction=P['prefilter_fraction']).fit(X,y)
            cols = screen.get_support(indices=True)
            diagnostics['prefilter_width'] = len(cols)
        score = forest(task, seed).fit(X[:, cols] if len(cols)<p else X, y).feature_importances_
        ranking = cols[np.argsort(-score, kind='stable')]
    elif method in {'exact','mean','k16'}:
        exact = ExactMinPI(task=task, random_state=seed).fit(X, y)
        if method == 'exact':
            weakest = exact.tier_rates_.argmin(axis=0)
            diagnostics.update(tiers=exact.tier_sizes_.tolist(), n_witnesses=exact.n_witnesses_,
                               first_argmin_counts=np.bincount(weakest,minlength=len(exact.tier_sizes_)).tolist(),
                               rates=exact.tier_rates_.tolist())
            ranking = exact.ranking_
        else:
            score = exact.tier_rates_.mean(axis=0) if method == 'mean' else exact.tier_rates_[-1]
            ranking = np.argsort(-score, kind='stable')
    else:
        margins = witness_margins(X, y, task, seed)
        if method == 'legacy_compact_rank':
            groups = len(np.unique(y)) if task == 'classification' else 5
            _, score, _ = _votes(margins, np.random.default_rng(seed), 'min')
            ranking = _diverse_order(margins, score, groups)
        else:
            votes = np.sign(margins).astype(np.int8)
            if method == 'margin':
                score = margins.mean(axis=0)
            elif method == 'vote':
                score = votes.mean(axis=0)
            elif method == 'mc32':
                tiers = np.unique(np.minimum(P['tiers'], p-1))
                score = mc_rates(votes, tiers, seed).min(axis=0)
                diagnostics['scores'] = score.tolist()
            else:
                raise KeyError(method)
            ranking = np.argsort(-score, kind='stable')
    return np.sort(ranking[:k]), diagnostics


def train(learner, X, y, Xt, task, seed):
    if learner == 'ridge':
        cls = RidgeClassifier if task == 'classification' else Ridge
        model = make_pipeline(StandardScaler(), cls(alpha=P['linear_alpha'], solver='lsqr',
                              tol=P['ridge_tol'], max_iter=P['ridge_max_iter']))
    else:
        model = forest(task, seed)
    model.fit(X, y)
    if task == 'regression':
        return model.predict(Xt), None
    if learner == 'ridge':
        score = model.decision_function(Xt)
        pred = model.classes_[(score > 0).astype(int)] if score.ndim == 1 else model.classes_[score.argmax(axis=1)]
    else:
        score = model.predict_proba(Xt)
        pred = model.classes_[score.argmax(axis=1)]
        if score.shape[1] == 2:
            score = score[:,1]
    return pred, score


def metrics(y, pred, score):
    if score is None:
        return {'r2':float(r2_score(y,pred)), 'mae':float(mean_absolute_error(y,pred))}
    auc = roc_auc_score(y,score) if score.ndim == 1 else np.mean([
        roc_auc_score(y==cls, score[:,cls]) for cls in range(score.shape[1])])
    return {'auc':float(auc), 'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred))}


def run_case(name, seed, pilot):
    family = 'pilot' if pilot else 'formal'
    parent = OUTPUT/family/name
    parent.mkdir(parents=True, exist_ok=True)
    finish = parent/f's{seed}'
    if finish.exists():
        raise FileExistsError(f'Case already exists: {finish}')
    X,y,task,names,colmap,truth,source_hash = load_dataset(name)
    a,b = train_test_split(np.arange(len(y)),test_size=P['test_fraction'],random_state=seed,
                           stratify=y if task=='classification' else None)
    train_x,test_x = X[a],X[b]
    del X
    ytr,yte = y[a],y[b]
    rng = np.random.default_rng(seed+4141)
    archive = dict(test_indices=b, train_indices=a, y_test=yte)
    entries = {}
    with tempfile.TemporaryDirectory(prefix='.minpi-',dir=parent) as scratch:
        build = Path(scratch)/finish.name
        build.mkdir()
        with threadpool_limits(limits=1):
            for arm in rng.permutation(P['arms']):
                start = perf_counter()
                chosen, diag = select(arm, train_x, ytr, task, seed)
                selection_seconds = perf_counter() - start
                # Persist high-width arrays separately, outside the timing interval.
                for key in ('rates','scores'):
                    if key in diag:
                        archive[f'{arm}_{key}'] = np.asarray(diag.pop(key))
                if len(chosen) != len(set(chosen)):
                    raise AssertionError('Duplicate selected columns')
                pair = {}
                for learner in rng.permutation(P['learners']):
                    start = perf_counter()
                    tx = train_x if len(chosen)==len(names) else train_x[:, chosen]
                    vx = test_x if len(chosen)==len(names) else test_x[:, chosen]
                    pred, score = train(learner, tx, ytr, vx, task, seed)
                    train_seconds = perf_counter() - start
                    archive[f'{arm}_{learner}_pred'] = pred
                    if score is not None:
                        archive[f'{arm}_{learner}_score'] = score
                    pair[learner] = dict(**metrics(yte,pred,score),backend_seconds=train_seconds,
                                         total_seconds=selection_seconds+train_seconds)
                entries[arm] = dict(selected=chosen.tolist(), k=len(chosen), selector_seconds=selection_seconds,
                                    backends=pair, diagnostics=diag)
        payload = dict(dataset=name, seed=seed, task=task, rows=len(y), p=len(names), train_rows=len(a),
                       test_rows=len(b), feature_names=names, original_column_indices=colmap.tolist(),
                       known_signal_original_ids=truth, dataset_source=source_hash, arms=entries,
                       versions=dict(python=sys.version, sklearn=sklearn.__version__, numpy=np.__version__,
                                     scipy=scipy.__version__, pandas=pd.__version__, platform=platform.platform()),
                       source_sha256={str(path.relative_to(ROOT).as_posix()):file_hash(path) for path in SOURCES})
        np.savez_compressed(build/'predictions.npz',**archive)
        (build/'record.json').write_text(json.dumps(payload,allow_nan=False,indent=2),encoding='utf-8')
        build.rename(finish)
    print(f'{family}/{name}/s{seed}: {len(entries)} arms x {len(P["learners"])} learners',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=P['datasets'],default=P['datasets'])
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    seeds=args.seeds or ([P['pilot_seed']] if args.pilot else P['formal_seeds'])
    allowed={P['pilot_seed']} if args.pilot else set(P['formal_seeds'])
    if not set(seeds)<=allowed:
        parser.error('Seed outside locked family')
    if args.child:
        if len(args.datasets)!=1 or len(seeds)!=1:
            parser.error('A child runs one dataset/seed')
        return run_case(args.datasets[0],seeds[0],args.pilot)
    family='pilot' if args.pilot else 'formal'
    for seed in seeds:
        for dataset in args.datasets:
            finish=OUTPUT/family/dataset/f's{seed}'
            if finish.exists():
                record=json.loads((finish/'record.json').read_text(encoding='utf-8'))
                if record['source_sha256']!={p.relative_to(ROOT).as_posix():file_hash(p) for p in SOURCES}:
                    raise RuntimeError(f'Source changed since {finish}; inspect before resuming')
                continue
            args_child=[sys.executable,'-B','-m','experiments.minpi_study','--child',
                        '--datasets',dataset,'--seeds',str(seed)]
            if args.pilot:
                args_child.append('--pilot')
            try:
                subprocess.run(args_child,check=True,timeout=P['timeout_per_case_seconds'],
                               env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
                failure=OUTPUT/family/dataset/f's{seed}-failure.json'
                failure.parent.mkdir(parents=True,exist_ok=True)
                failure.write_text(json.dumps(dict(dataset=dataset,seed=seed,error=str(error))),encoding='utf-8')
                raise


if __name__=='__main__':
    main()

