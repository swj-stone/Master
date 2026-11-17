"""Compare refitted and stopped RFs under one common law on the original inputs."""
import argparse
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import quota_coalition_rf as rf, validation_predictive as seven
from experiments.minpi_rework import wide_veto_data as wide
from experiments.minpi_rework.predictive_mask_study import centered
from experiments.minpi_rework.predictive_selection import prepare_paths, select_width
from minppi_mofs.availability_minpi import probabilities
from minppi_mofs.predictive_quota import PredictiveGame


OUT = b.ROOT / "thesis-paper/evidence/minpi_rework/predictive-refit"
PROTOCOL = Path(__file__).with_name("predictive_refit_protocol.md")
PANEL = ([("seven", name, seed) for name in seven.SOURCES for seed in seven.SEEDS] +
         [("wide", name, 42) for name in wide.SOURCES])


def grid(p, drop):
    sizes = [p]
    while sizes[-1] > 1:
        k = sizes[-1]
        sizes.append(k - max(1, int(k * drop)))
    return sizes


def risk(model, x, y, task, ambient, selected):
    """Expected (unnormalized) squared loss at every original availability size."""
    game = model.game(x)
    rows = centered(model, y, task)
    criterion = PredictiveGame.from_paths(game.a, game.incidence, rows)
    empty = np.zeros(game.p, bool)
    gain = criterion.coefficients(empty) - criterion.coefficients(selected)
    return criterion.null - probabilities(ambient, len(gain)-1) @ gain


def normalized_worst(loss, reference):
    p = len(reference) - 1
    return float(np.max((loss[1:] - reference[1:]) * (p / np.arange(1, p+1)))
                 / max(1e-100, reference[0]))


def check():
    rng = np.random.default_rng(481)
    x = rng.normal(size=(94, 6))
    target = x[:, 0] - 2*x[:, 2] + x[:, 3]
    for task, y in (("classification", (target > 0).astype(int)), ("regression", target)):
        parent = rf.fit(x[:60], y[:60], task, 17, n_estimators=7, max_depth=3)
        subset = np.array([0, 2, 4])
        refitted = rf.fit(x[:60, subset], y[:60], task, 311, n_estimators=9, max_depth=3)
        test, truth = x[60:], y[60:]
        observed = np.eye(2)[truth] if task == "classification" else truth[:, None]
        for model, columns in ((parent, np.arange(6)), (refitted, subset)):
            for keep in (np.ones(len(columns),bool), np.arange(len(columns)) % 2 == 0):
                curve = risk(model, test[:, columns], truth, task, 6, keep)
                for s in range(7):
                    losses = []
                    for mask in combinations(range(6), s):
                        ids = np.flatnonzero(keep & np.isin(columns, mask))
                        pred = model.predict(test[:, columns[ids]], ids).reshape(observed.shape)
                        losses.append(np.square(observed - pred).sum())
                    np.testing.assert_allclose(curve[s], np.mean(losses), rtol=1e-11, atol=1e-10)
    print("All exact cross-forest risks matched direct masks for both learners and support types.")


def mask(p, cols):
    selected = np.zeros(p, dtype=bool)
    selected[cols] = True
    return selected


def deploy(parent, model, x, cols, refitted):
    return model.predict(x[:, cols]) if refitted else parent.predict(x[:, cols], cols)


def collect_one(panel, name, seed):
    provider = seven if panel == "seven" else wide
    x, y, task, idx = provider.split(name, seed)
    if not (np.isnan(y[idx["test"]]).all() if task == "regression" else (y[idx["test"]] == -1).all()):
        raise ValueError("Unsealed test outcomes")
    xf, yf = x[idx["fit"]], y[idx["fit"]]
    xw, yw = x[idx["width"]], y[idx["width"]]
    p = x.shape[1]
    scale = max(1e-12, np.std(yf))
    tick = perf_counter()
    parent = rf.fit(xf, yf, task, seed)
    fit_s = perf_counter() - tick
    paths, null, costs = prepare_paths(parent, x[idx["vote"]], y[idx["vote"]], xw, yw, task)
    stopped = {}
    for arm, path in (("mdi", {"mdi": paths["mdi"]}), ("minpi", paths)):
        tick = perf_counter()
        cols, trace = select_width(path, null, yw, task, scale)
        basic = costs["calibration"] + costs["mdi_rank"] + costs["mdi_curve"]
        extra = (costs["min_rank"] + costs["min_curve"]) if arm == "minpi" else 0.
        stopped[arm] = dict(support=cols.tolist(), cal=trace["worst"], refitted=False,
                            seconds=fit_s + basic + extra + perf_counter() - tick, fits=1)
    tick = perf_counter()
    all_cols = np.arange(p)
    reference = risk(parent, xw, yw, task, p, np.ones(p, bool))
    native = b.metrics(yw, parent.native(xw), task, scale)
    ref_setup = perf_counter() - tick
    mdi = paths["mdi"].order
    fits = {}
    candidates = {}

    def try_refit(cols):
        key = tuple(cols)
        if key in candidates:
            return candidates[key]
        if len(cols) == 0:
            raise ValueError("Cannot refit a forest on zero columns")
        start = perf_counter()
        model = parent if len(cols) == p else rf.fit(xf[:, cols], yf, task, seed)
        pred = model.native(xw[:, cols])
        ok = b.eligible(b.metrics(yw, pred, task, scale), native, task)
        vote = None
        if ok:
            curve = reference if len(cols) == p else risk(model, xw[:,cols], yw, task, p,
                                                          np.ones(len(cols), bool))
            vote = normalized_worst(curve, reference)
        result = dict(support=list(map(int, cols)), cal=vote, refitted=len(cols) < p,
                      seconds=perf_counter()-start, fits=int(len(cols) < p))
        fits[key] = model
        candidates[key] = result
        return result

    policies = dict(stopped)
    for arm, fraction in (("refit20", .2), ("refit50", .5)):
        widths = grid(p, fraction)
        trials = [try_refit(np.sort(mdi[p-k:])) for k in widths]
        choice = min((v for v in trials if v["cal"] is not None and v["cal"] <= .02 + 1e-12),
                     key=lambda v:(len(v["support"]), v["cal"]))
        policies[arm] = dict(choice, seconds=fit_s + ref_setup + costs["mdi_rank"]
                             + sum(v["seconds"] for v in trials), fits=1 + sum(v["fits"] for v in trials))
    for arm, stop in stopped.items():
        if not stop["support"]:
            policies[arm+"_one_refit"] = dict(stop)
            continue
        candidate = try_refit(np.asarray(stop["support"], int))
        improves = candidate["cal"] is not None and candidate["cal"] <= .02 + 1e-12 and candidate["cal"] < stop["cal"]
        choice = candidate if improves else stop
        policies[arm+"_one_refit"] = dict(choice, seconds=stop["seconds"]+ref_setup+candidate["seconds"],
                                          fits=stop["fits"] + candidate["fits"])
    policies["full"] = dict(support=all_cols.tolist(), cal=0., refitted=False, seconds=fit_s, fits=1)
    for item in policies.values():
        cols = np.array(item["support"], int)
        model = fits[tuple(cols)] if item["refitted"] else parent
        prediction = deploy(parent, model, x[idx["test"]], cols, item["refitted"])
        item["forecast_sha"] = sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest()
    return policies


def score_one(panel, name, seed, choices):
    x, y, task, idx = seven._partition(name, seed) if panel == "seven" else wide.partition(name, seed)
    fit, test = idx["fit"], idx["test"]
    p = x.shape[1]
    parent = rf.fit(x[fit], y[fit], task, seed)
    full_risk = risk(parent, x[test], y[test], task, p, np.ones(p,bool))
    scale = max(1e-12, np.std(y[np.r_[fit, idx["vote"], idx["width"]]]))
    models = {}
    rows = []
    for arm, choice in choices.items():
        cols = np.array(choice["support"], int)
        key = tuple(cols)
        refitted = choice["refitted"]
        if refitted and key not in models:
            models[key] = rf.fit(x[np.ix_(fit,cols)], y[fit], task, seed)
        model = models[key] if refitted else parent
        pred = deploy(parent, model, x[test], cols, refitted)
        if sha256(np.ascontiguousarray(pred).tobytes()).hexdigest() != choice["forecast_sha"]:
            raise ValueError("Sealed forecast did not reproduce")
        losses = risk(model, x[np.ix_(test,cols)] if refitted else x[test], y[test], task, p,
                      np.ones(len(cols),bool) if refitted else mask(p,cols))
        rows.append(dict(panel=panel,dataset=name,seed=seed,task=task,method=arm,p=p,k=len(cols),
                         outer_worst=normalized_worst(losses,full_risk),
                         **{k:choice[k] for k in ("refitted","cal","seconds","fits")},
                         **b.metrics(y[test],pred,task,scale)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "collect", "score"))
    args = parser.parse_args()
    if args.phase == "check":
        return check()
    OUT.mkdir(parents=True, exist_ok=True)
    lockpath = OUT / "lock.json"
    if args.phase == "collect":
        choices = {}
        for i, (panel, name, seed) in enumerate(PANEL, 1):
            timer = Timer(3600, lambda: os._exit(124))
            timer.daemon = True
            timer.start()
            start = perf_counter()
            try:
                choices[f"{panel}/{name}/{seed}"] = collect_one(panel, name, seed)
            finally:
                timer.cancel()
            print(f"[{i}/{len(PANEL)}] {name}/{seed} {perf_counter()-start:.1f}s", flush=True)
        lockpath.write_text(json.dumps(dict(protocol=b.digest(PROTOCOL), code=b.digest(Path(__file__)),
                                            choices=choices), indent=2), encoding="utf8")
    else:
        lock = json.loads(lockpath.read_bytes())
        if (lock["protocol"] != b.digest(PROTOCOL) or lock["code"] != b.digest(Path(__file__))
                or len(lock["choices"]) != len(PANEL)):
            raise ValueError("Incomplete or obsolete refit forecast lock")
        rows = []
        for panel, name, seed in PANEL:
            rows.extend(score_one(panel,name,seed,lock["choices"][f"{panel}/{name}/{seed}"]))
        pd.DataFrame(rows).to_csv(OUT/"scores.csv",index=False)


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
