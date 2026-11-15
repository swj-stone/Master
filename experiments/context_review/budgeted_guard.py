"""Compare equal guard-refit budgets on the same feature sets before opening outer outcomes."""
import argparse
from itertools import combinations
import json
from math import comb
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from urllib.request import urlopen
from shutil import copyfileobj
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiments.context_review.refit_target import metrics
from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import DATA, OUT as PRIMARY, ROOT, config, digest
from experiments.context_study.working_set import oracle
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.piv3_compact_tree import select as gain_anchor
from minppi_mofs.piv3_compact_witness import _sample, _within

DESIGN = ROOT / "configs/context_review/budgeted_guard.json"
RUN = ROOT / "results/context_review/budgeted-guard"
DONOR = ROOT / "results/context_review/refit-guard"
ARCHIVE = ROOT / "data/context_study/uci-optdigits/uci80.zip"
URL = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
SHA = "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0"
METHODS = ("fraction_gain", "vote_gain", "fraction_lottery", "vote_lottery")


def read(path):
    return json.loads(path.read_bytes())


def digits(columns, member):
    with ZipFile(ARCHIVE) as archive, archive.open(member) as handle:
        return pd.read_csv(handle, header=None, usecols=columns)


def digit_target():
    return digits([64], "optdigits.tes").iloc[:, 0].to_numpy(dtype=int)


def inputs(name, fold):
    if name == "optdigits":
        if not ARCHIVE.exists():
            ARCHIVE.parent.mkdir(exist_ok=True, parents=True)
            download = ARCHIVE.with_suffix(".partial")
            with urlopen(URL, timeout=90) as source, download.open("wb") as target:
                copyfileobj(source, target)
            if digest(download) != SHA:
                raise ValueError("Different UCI archive")
            download.replace(ARCHIVE)
        assert digest(ARCHIVE) == SHA
        training = digits(range(65), "optdigits.tra")
        y = training.pop(64).to_numpy(dtype=int)
        x = training.to_numpy(dtype=np.float32)
        future = digits(range(64), "optdigits.tes").to_numpy(dtype=np.float32)
        assert x.shape == (3823, 64) and future.shape == (1797, 64)
        train, rank = train_test_split(np.arange(len(x)), test_size=.25, random_state=20421, stratify=y)
        prop, guard = train_test_split(rank, test_size=.5, random_state=39823, stratify=y[rank])
        return x, y, train, prop, guard, np.arange(len(x)), future, digit_target, None, SHA
    src = PRIMARY / name / f"fold-{fold}"
    donor = DONOR / name / f"fold-{fold}"
    receipt = read(src / "completed.json")
    split = read(src / "selection.json")
    layout = read(donor / "proposals.json")
    assert receipt["selection_sha256"] == digest(src / "selection.json") == layout["donor"]
    assert read(donor / "selection.json")["proposals_sha256"] == digest(donor / "proposals.json")
    table = DATA / name / f"{name}.tsv.gz"
    assert receipt["data_sha256"] == digest(table)
    data = pd.read_csv(table, sep="\t")
    y = data.pop("target").map({c: i for i, c in enumerate(split["class_order"])}).to_numpy()
    x = data.to_numpy(dtype=np.float32)
    train, prop, guard, test = (np.array(layout[k]) for k in ("train", "proposal", "guard", "test"))
    dev = np.sort(np.concatenate([train, prop, guard]))
    assert set(dev).isdisjoint(test) and len(dev) + len(test) == len(x)
    return x, y, train, prop, guard, dev, x[test], lambda: y[test], layout, receipt["data_sha256"]


def candidate_pool(order, width, index, fold):
    anchor = sorted(order)
    prefix = tuple(sorted(order[:width]))
    count = comb(len(anchor), width)
    if count <= 8:
        sets = sorted(combinations(anchor, width))
    else:
        rng = np.random.default_rng([81027, index, fold, width])
        candidates = {prefix}
        while len(candidates) < 8:
            draw = tuple(sorted(map(int, rng.choice(anchor, size=width, replace=False))))
            candidates.add(draw)
        sets = sorted(candidates)
    rng = np.random.default_rng([72291, index, fold, width])
    lottery = [sets[i] for i in rng.permutation(len(sets))]
    gain_pos = {col: i for i, col in enumerate(order)}
    gain = sorted(sets, key=lambda s: (sum(gain_pos[col] for col in s), s))
    assert gain[0] == prefix
    return sets, gain, lottery


def mask_minima(model, train, proposal, labels, ordered, anchor_k, seed, pool, widths):
    start = perf_counter()
    if not widths:
        return {}, 0, 0.
    task = "binary" if model.n_classes_ == 2 else "multiclass"
    ids, weights = _sample(labels, task, np.random.SeedSequence([seed, 659]))
    rare = task == "binary" and min(np.bincount(labels)) <= .05*len(labels)
    bg = np.random.default_rng(seed).choice(len(train), min(len(train), 512), replace=False)
    anchor = frozenset(ordered[:anchor_k])
    ballot, cache = oracle(model, proposal[ids], np.median(train[bg], axis=0), labels[ids], weights, anchor, rare)
    family = separating_contexts(ordered[anchor_k:])
    scores = {}
    for size in widths:
        for cols in pool[size][0]:
            lowest = 4
            for extension in family:
                lowest = min(lowest, sum(ballot(frozenset(cols), extension)))
                if lowest == 0:
                    break
            scores[cols] = int(lowest)
    return scores, len(cache), perf_counter() - start


def predict_sets(x, y, train, x_eval, supports, seed):
    start = perf_counter()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    a, b = imputer.transform(x[train]), imputer.transform(x_eval)
    prep = perf_counter() - start
    probs, cost = {}, {}
    for cols in sorted(set(supports)):
        start = perf_counter()
        probs[cols] = fit(a[:, cols], y[train], seed).predict_proba(b[:, cols])
        cost[cols] = perf_counter() - start
    return probs, cost, prep


def run_fold(name, fold, index):
    dest = RUN / name / f"fold-{fold}"
    dest.mkdir(exist_ok=True, parents=True)
    x, y, train, prop, guard, dev, future, get_target, donor, source_hash = inputs(name, fold)
    seed = 20421 + fold
    start = perf_counter()
    prep = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    xt, xp = prep.transform(x[train]), prep.transform(x[prop])
    model = fit(xt, y[train], seed)
    task = "binary" if model.n_classes_ == 2 else "multiclass"
    names = [f"c{i}" for i in range(x.shape[1])]
    _, info = gain_anchor(model, pd.DataFrame(xt, columns=names), y[train], pd.DataFrame(xp, columns=names),
                          y[prop], task=task, utility="ovr_auc", seed=seed, aggregate="gain")
    order = [int(c[1:]) for c in info["order"]]
    k = int(info["k"])
    anchor = tuple(sorted(order[:k]))
    if donor:
        assert order == donor["order"] and list(anchor) == donor["candidates"]["anchor"]
    half = (k + 1)//2
    widths = sorted({(j*k+3)//4 for j in (1, 2, 3)} - {k})
    anchor_s = perf_counter() - start
    pool, pool_s = {}, {}
    for w in widths:
        start = perf_counter()
        pool[w] = candidate_pool(order[:k], w, index, fold)
        pool_s[w] = perf_counter() - start
    budgets = {"one": [half] if half < k else [], "three": widths}
    slates, screens, screening_s, questions = {}, {}, {}, {}
    for budget, grid in budgets.items():
        scores, mask_calls, screen_s = mask_minima(model, xt, xp, y[prop], order, k, seed, pool, grid)
        screens[budget] = [dict(cols=s, approvals=n) for s, n in scores.items()]
        screening_s[budget], questions[budget] = screen_s, mask_calls
        for method in METHODS:
            selected = []
            for w in grid:
                tie = pool[w][1 if method.endswith("gain") else 2]
                nominee = max(tie, key=scores.__getitem__) if method.startswith("vote") else tie[0]
                selected.append(nominee)
            slates[f"{budget}_{method}"] = selected
    store(dest/"shortlists.json", dict(source=source_hash, order=order, anchor=anchor, pools=pool,
          screens=screens, shortlists=slates, train=train.tolist(), proposal=prop.tolist(), guard=guard.tolist(),
          dev=dev.tolist(), seed=seed, protocol=digest(RUN/"protocol.json")))
    candidates = {anchor}
    candidates.update(s for shortlist in slates.values() for s in shortlist)
    inner = np.sort(np.concatenate([train, prop]))
    p, guard_cost, guard_prep = predict_sets(x, y, inner, x[guard], candidates, seed)
    rare = bool(task == "binary" and min(np.bincount(y[prop])) <= .05*len(prop))
    m = {s:metrics(y[guard], v, rare) for s, v in p.items()}
    passed = {s:bool(_within(val, m[anchor])) for s, val in m.items()}
    final = {arm:next((s for s in ss if passed[s]), anchor) for arm, ss in slates.items()}
    final["anchor"] = anchor
    supports = sorted(candidates)
    np.savez_compressed(dest/"guard.npz", y=y[guard], **{f"s{i}":p[s] for i, s in enumerate(supports)})
    store(dest/"selection.json", dict(shortlists=digest(dest/"shortlists.json"), guard_sha=digest(dest/"guard.npz"),
          supports=supports, guard_pass=[passed[s] for s in supports], selections=final, rare=rare))
    frozen = digest(dest/"selection.json")
    p_final, final_cost, final_prep = predict_sets(x, y, dev, future, final.values(), seed)
    used = sorted(p_final)
    np.savez_compressed(dest/"predictions.npz", **{f"s{i}":p_final[s] for i,s in enumerate(used)})
    predictions_hash = digest(dest/"predictions.npz")
    assert digest(dest/"selection.json") == frozen
    # External Optdigits labels are first read here, after the prediction arrays are saved.
    test_y = get_target()
    results = {s:metrics(test_y, v, rare) for s, v in p_final.items()}
    rows = []
    for arm, s in final.items():
        fail = not _within(results[s], results[anchor])
        budget, _, method = arm.partition("_")
        shortlist = slates.get(arm, [])
        grid = budgets.get(budget, [])
        masked = method.startswith("vote")
        common_s = anchor_s + sum(pool_s[w] for w in grid)
        guard_s = guard_prep + sum(guard_cost[c] for c in {anchor, *shortlist})
        outer_s = final_prep + final_cost[s]
        screen_s = screening_s[budget] if masked else 0.
        rows.append(dict(dataset=name, fold=fold, budget=budget, method=method or "anchor",
                    k=len(s), anchor_k=k, columns=x.shape[1], failure=fail, safe_saved=(1-len(s)/k)*(not fail),
                    guard_candidates=len(shortlist), shortlist_indices=[supports.index(c) for c in shortlist],
                    selected_index=supports.index(s), masks=questions[budget] if masked else 0,
                    anchor_pool_s=common_s, vote_s=screen_s, guard_s=guard_s, final_s=outer_s,
                    total_s=common_s + screen_s + guard_s + outer_s, **results[s]))
    store(dest/"completed.json", dict(rows=rows, selection=frozen, predictions=predictions_hash,
          unique_guard_models=len(candidates), unique_outer_models=len(used), n_test=len(test_y)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3, metavar=("TASK", "FOLD", "INDEX"))
    args = parser.parse_args()
    if args.worker:
        name, fold, index = args.worker
        with threadpool_limits(limits=1):
            run_fold(name, int(fold), int(index))
        return
    RUN.mkdir(exist_ok=True, parents=True)
    files = [DESIGN, Path(__file__), DONOR/"eligibility.csv", DONOR/"protocol.json",
             ROOT/"experiments/context_study/working_set.py", ROOT/"experiments/context_study/benchmark.py",
             ROOT/"src/minppi_mofs/piv3_compact_tree.py",ROOT/"src/minppi_mofs/piv3_compact_witness.py"]
    lock = {p.relative_to(ROOT).as_posix():digest(p) for p in files}
    path = RUN/"protocol.json"
    if path.exists() and read(path) != lock:
        raise ValueError("Frozen budgeted-guard sources changed")
    store(path, lock)
    tasks = config()["datasets"]
    jobs = [(r.dataset, r.fold, tasks.index(r.dataset)) for r in pd.read_csv(DONOR/"eligibility.csv").itertuples() if r.eligible]
    assert len(jobs) == 41
    jobs.insert(0, ("optdigits", 0, 100))
    rows = []
    start = perf_counter()
    for i, (name, fold, index) in enumerate(jobs, 1):
        dest = RUN / name / f"fold-{fold}"
        if not (dest/"completed.json").exists():
            subprocess.run([sys.executable, "-m", "experiments.context_review.budgeted_guard", "--worker",
                            name, str(fold), str(index)], check=True, timeout=3600)
        rec = read(dest/"completed.json")
        assert read(dest/"shortlists.json")["protocol"] == digest(path)
        assert rec["predictions"] == digest(dest/"predictions.npz") and rec["selection"] == digest(dest/"selection.json")
        rows.extend(rec["rows"])
        elapsed = perf_counter() - start
        print(f"[{i}/{len(jobs)}] {name} fold {fold}, elapsed {elapsed:.1f}s, remaining ~{elapsed*(len(jobs)-i)/i:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RUN/"per-fold.csv", index=False)


if __name__ == "__main__":
    main()
