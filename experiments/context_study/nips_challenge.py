"""Published NIPS 2003 validation splits, sparse model adapter and frozen-subset evaluation."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
import warnings
from zipfile import ZipFile

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from threadpoolctl import threadpool_limits

from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.context_design import matched_random_contexts, separating_contexts
from minppi_mofs.context_vote import search
from minppi_mofs.piv3_compact_witness import _sample, _within


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/context_study/nips2003.json"
RAW = ROOT / "data/raw/nips2003"
OUT = ROOT / "results/context_study/nips2003"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8")


def member(archive, dataset, suffix):
    name = f"{dataset}_{suffix}"
    matches = [k for k in archive.namelist() if k.rsplit("/", 1)[-1] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one official {name} member")
    return archive.read(matches[0])


def matrix(data, dataset, width):
    if dataset == "arcene":
        result = sparse.csr_matrix(np.loadtxt(io.BytesIO(data), dtype=float))
        if result.shape[1] != width:
            raise ValueError("Wrong Arcene feature count")
        return result
    indices, values, ptr = [], [], [0]
    for line in data.splitlines():
        if dataset == "dexter":
            for token in line.split():
                col, val = token.split(b":")
                indices.append(int(col) - 1)
                values.append(float(val))
        else:
            cols = [int(c) - 1 for c in line.split()]
            indices.extend(cols)
            values.extend([1.] * len(cols))
        ptr.append(len(indices))
    if min(indices) < 0 or max(indices) >= width:
        raise ValueError(f"Invalid feature index in {dataset}")
    result = sparse.csr_matrix((values, indices, ptr), shape=(len(ptr)-1, width))
    result.sort_indices()
    return result


def load_train_inputs(name, width):
    path = RAW / f"{name}.zip"
    hashes = {}
    with ZipFile(path) as z:
        blobs = {}
        for suffix in ("train.data", "valid.data", "train.labels"):
            blob = member(z, name, suffix)
            blobs[suffix] = blob
            hashes[suffix] = hashlib.sha256(blob).hexdigest()
    y = (np.loadtxt(io.BytesIO(blobs["train.labels"])) > 0).astype(int)
    x = matrix(blobs["train.data"], name, width)
    xv = matrix(blobs["valid.data"], name, width)
    if len(y) != x.shape[0] or set(y) != {0, 1}:
        raise ValueError(f"Invalid training labels for {name}")
    return x, y, xv, hashes


def model(seed, penalty="l2", c=1.):
    return LogisticRegression(C=c, solver="liblinear", penalty=penalty, dual=penalty=="l2",
                              max_iter=2000, tol=1e-5, random_state=seed)


def probability(logit):
    p = expit(logit)
    return np.column_stack((1-p, p))


def common_fit(x, y, train, rank, seed):
    start = perf_counter()
    scale = MaxAbsScaler().fit(x[train])
    xi, xr = scale.transform(x[train]), scale.transform(x[rank])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.filterwarnings("ignore", message="Features .* are constant")
        stat, _ = f_classif(xi, y[train])
    order = np.argsort(-np.where(np.isnan(stat), -np.inf, stat), kind="stable")
    screen = order[~np.isnan(stat[order])][:256]
    screen_done = perf_counter()
    base = model(seed).fit(xi[:, screen], y[train])
    coef = base.coef_.ravel()
    order = np.argsort(-np.abs(coef), kind="stable").tolist()
    fill = np.median(xi[:, screen].toarray(), axis=0)
    # For a linear logit, replacing omitted coordinates by their medians is exact.
    center = base.intercept_[0] + np.dot(coef, fill)
    contributions = (xr[:, screen].toarray()-fill)*coef
    base_scores = make_scorer(y[rank], [np.ones(len(rank))], "binary", "roc_auc", False)
    logits = np.full(len(rank), center)
    full = base_scores(probability(center + contributions.sum(axis=1)))[0]
    for k, j in enumerate(order, 1):
        logits += contributions[:, j]
        scores = base_scores(probability(logits))[0]
        if all(full[m]-scores[m] <= .08*abs(full[m])+1e-12 for m in full):
            break
    return dict(screen=screen, order=order, k=k, matrix=xi[:, screen], rank=xr[:, screen],
                center=center, contributions=contributions, screen_s=screen_done-start,
                anchor_s=perf_counter()-screen_done)


def vote_run(common, yr, seed, rule, contexts):
    order, k = common["order"], common["k"]
    anchor = frozenset(order[:k])
    rows, weights = _sample(yr, "binary", np.random.SeedSequence([seed, 659]))
    point = rule == "point"
    scorer = make_scorer(yr[rows], weights[:1] if point else weights, "binary", "roc_auc", False)
    contrib = common["contributions"][rows]
    cache = {}
    def assess(mask):
        if mask not in cache:
            logits = common["center"] + contrib[:, sorted(mask)].sum(axis=1)
            cache[mask] = scorer(probability(logits))
        return cache[mask]
    def ballot(s, e):
        yes = tuple(_within(a,b) for a,b in zip(assess(s|e), assess(anchor|e), strict=True))
        return yes*4 if point else yes
    live, trace = search(order[:k], contexts, ballot, rule="point" if point else "min")
    return sorted(live), len(cache), len(trace["trace"])


def l1_choice(common, y, train, rank, seed):
    scored = []
    for c in (.01, .1, 1., 10.):
        classifier = model(seed, "l1", c).fit(common["matrix"], y[train])
        kept = np.flatnonzero(classifier.coef_.ravel()).tolist()
        if kept:
            refit = model(seed).fit(common["matrix"][:, kept], y[train])
            auc = roc_auc_score(y[rank], refit.decision_function(common["rank"][:, kept]))
            scored.append((auc, -len(kept), -c, kept))
    return max(scored)[3] if scored else common["order"][:1]


def freeze_split(x, xv, y, name, seed, methods, folder):
    start = perf_counter()
    train, rank = train_test_split(np.arange(len(y)), test_size=.30, stratify=y, random_state=seed)
    common = common_fit(x,y,train,rank,seed)
    screen, order, k = common["screen"], common["order"], common["k"]
    tail = order[k:]
    nested = ((), tail[:k], tail)
    codes = separating_contexts(tail)
    random_masks = matched_random_contexts(codes, tail, seed=31071+seed)
    families = dict(nested=nested, point=((),), separating=codes, random_matched=random_masks)
    rows, predictions = [], {}
    supports = {}
    for arm in methods:
        step = perf_counter()
        calls = trials = 0
        if arm == "raw_full":
            support = np.arange(x.shape[1])
        elif arm == "screen_full":
            support = screen
        elif arm == "anchor":
            support = screen[order[:k]]
        elif arm == "prefix_at_separating_k":
            support = screen[order[:len(supports["separating"])]]
        elif arm == "l1":
            support = screen[l1_choice(common, y, train, rank, seed)]
        else:
            idx, calls, trials = vote_run(common, y[rank], seed, arm, families[arm])
            support = screen[idx]
        supports[arm] = sorted(map(int,support))
        selection = perf_counter() - step
        # The official validation outcomes are read only in the later scoring pass.
        refit_start = perf_counter()
        xx = x[:, support]
        scaling = MaxAbsScaler().fit(xx)
        estimator = model(seed).fit(scaling.transform(xx), y)
        pred = estimator.predict_proba(scaling.transform(xv[:, support]))[:,1]
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite predictions")
        predictions[arm] = pred
        rows.append(dict(dataset=name, seed=seed, arm=arm, width=x.shape[1], screen=len(screen),
                         anchor=k, k=len(support), mask_queries=calls, proposals=trials,
                         screen_s=common["screen_s"] if arm!="raw_full" else 0.,
                         anchor_s=common["anchor_s"] if arm in {*families,"anchor","prefix_at_separating_k"} else 0.,
                         selection_s=selection, refit_s=perf_counter()-refit_start))
        if perf_counter()-start > 3600:
            raise TimeoutError(f"{name}/{seed} exceeded one hour")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/"predictions.npz", **predictions)
    # Full raw support has a canonical compact representation.
    supports["raw_full"] = {"all_columns":x.shape[1]}
    write_json(folder/"frozen.json", dict(supports=supports,train=train.tolist(),rank=rank.tolist(),rows=rows))
    return rows


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_bytes())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol":digest(CONFIG), "runner":digest(Path(__file__)),
                "sources":{n:digest(RAW/f"{n}.zip") for n in config["datasets"]}}
    for module in ("context_design", "context_vote", "piv3_compact_witness"):
        manifest[module] = digest(ROOT/"src/minppi_mofs"/f"{module}.py")
    manifest["scoring"] = digest(ROOT/"experiments/test/minpi_convergence_metrics.py")
    path = args.out/"protocol.json"
    if path.exists() and json.loads(path.read_bytes()) != manifest:
        raise ValueError("A specification or input changed. Use a new output directory.")
    write_json(path,manifest)
    rows = []
    for name, (uci,width) in config["datasets"].items():
        x, y, xv, hashes = load_train_inputs(name,width)
        for seed in config["seeds"]:
            folder = args.out/name/str(seed)
            if not (folder/"frozen.json").exists():
                begin = perf_counter()
                freeze_split(x,xv,y,name,seed,config["methods"],folder)
                print(f"{name} seed {seed}: all supports frozen in {perf_counter()-begin:.1f}s",flush=True)
        with ZipFile(RAW/f"{name}.zip") as z:
            labels = member(z,name,"valid.labels")
        hashes["valid.labels"] = hashlib.sha256(labels).hexdigest()
        write_json(args.out/name/"members.json",hashes)
        yv = (np.loadtxt(io.BytesIO(labels))>0).astype(int)
        for seed in config["seeds"]:
            folder = args.out/name/str(seed)
            with np.load(folder/"predictions.npz") as pred:
                for record in json.loads((folder/"frozen.json").read_bytes())["rows"]:
                    p = pred[record["arm"]]
                    record.update(auc=roc_auc_score(yv,p),accuracy=accuracy_score(yv,p>.5),
                                  ber=1-balanced_accuracy_score(yv,p>.5))
                    rows.append(record)
    with (args.out/"scores.csv").open("w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed {len(rows)} arm records",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run()
