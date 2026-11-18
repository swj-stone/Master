"""Compare the frozen half- and three-quarter-width moment minPI against matched controls."""
import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import budget_minpi_deployments as deployment
from experiments import fast_minpi_five_source as five
from experiments.compare_predictive_minpi import choose
from experiments.stable_minpi_stress import (
    DATASETS, ROOT, SPLIT_SEED, digest, file_hash, write_json,
)
from minppi_mofs.moment_minpi import moment_top_from_training
from minppi_mofs.sketch_minpi import reduced_width
from minppi_mofs.stable_minpi import _impute_validation


BUDGETS = {50: .5, 75: .75}
CANDIDATES = tuple(f"moment_{k}" for k in BUDGETS)
METHODS = tuple(f"{kind}_{k}" for k in BUDGETS for kind in ("moment", "F", "random")) + ("full",)
EARLIER_SCORE_FILES = [deployment.RUNS / run / "cases.csv"
                       for run in ("fast-five-20261117a", "fast-deployments-20261117a")]
DEPLOYMENT_INPUTS = EARLIER_SCORE_FILES[1].with_name("inputs.csv")


@dataclass
class Panel:
    name: str
    fold: int
    seed: int
    X: np.ndarray
    y: np.ndarray
    Xt: np.ndarray
    gold: np.ndarray
    task: str
    groups: np.ndarray | None
    units: np.ndarray | None
    historical_full: np.ndarray


def panels():
    for name, fold, X, y, Xt, gold, task, groups in five.data_folds():
        with np.load(five.CONSENSUS_ARCHIVE/f"{name}-{fold}.npz") as old:
            for seed in five.SEEDS:
                yield Panel(name, fold, seed, X, y, Xt, gold, task, groups, None, old[f"full_s{seed}_pred"])
    for name in deployment.SOURCES:
        for i, data in enumerate(deployment.configurations(name), 1):
            seed, X, y, groups, Xt, gold, units, task, extra = data
            full, _, _ = deployment.prior_arrays(name, seed, gold, units, extra)
            fold = 1 if name in deployment.REPEATED_OUTER else i
            yield Panel(name, fold, seed, X, y, Xt, gold, task, groups, units, full)


def select(method, panel):
    p = panel.X.shape[1]
    if method == "full":
        return np.arange(p), {}
    kind, budget = method.split("_")
    count = reduced_width(p, BUDGETS[int(budget)])
    if kind == "moment":
        return moment_top_from_training(
            panel.X, panel.y, count, task=panel.task, random_state=panel.seed,
            groups=panel.groups, ordered_groups=panel.name in ("gas", "secom"),
        )
    X = panel.X
    if kind == "F" and panel.name in deployment.ADDITIONAL:
        X, _ = _impute_validation(X, X[:0])
    return choose(kind.lower(), X, panel.y, panel.task, panel.seed, count), {}


def protocol(repeats):
    prev_five = five.protocol(repeats)
    prev_deployment = deployment.protocol(repeats)
    def canonical(data):
        return {Path(k).as_posix(): v for k,v in data.items()}
    sources = canonical(prev_five["sources"]) | prev_deployment["source_sha256"]
    sources.update({p.relative_to(ROOT).as_posix(): file_hash(p) for p in
                    (Path(__file__), ROOT/"src/minppi_mofs/moment_minpi.py")})
    archives = canonical(prev_five["archives"]) | prev_deployment["archived_sha256"]
    archives.update({f.relative_to(ROOT).as_posix(): file_hash(f) for f in [*EARLIER_SCORE_FILES, DEPLOYMENT_INPUTS]})
    return dict(
        source_sha256=sources, archives=archives,
        platform=prev_deployment["platform"], cpu=prev_deployment["cpu"], python=prev_deployment["python"],
        numpy=prev_deployment["numpy"], sklearn=prev_deployment["sklearn"], threads=1, repeats=repeats,
        cases="45 five-source archived stress cases and 24 archived deployment cases. Known development "
              "data; three seeds can share a held-out population.", seeds=list(five.SEEDS),
        datasets=[*DATASETS, *deployment.SOURCES], budgets=BUDGETS, methods=list(METHODS),
        primary="moment_50 fixed before testing", sensitivity="moment_75 fixed before testing",
        score=prev_deployment["score"], split=prev_deployment["split"], timing=prev_deployment["timing"],
        success=prev_deployment["success"],
    )


def run_panel(panel, repeats, verifying=None):
    name, fold, seed = panel.name, panel.fold, panel.seed
    cache, quality, units, audit = {}, {}, [], []
    for method in (METHODS if verifying is None else CANDIDATES):
        cols, info = select(method, panel)
        five.check_columns(cols, panel.X.shape[1])
        if method != "full":
            count = reduced_width(panel.X.shape[1], BUDGETS[int(method.split("_")[1])])
            assert len(cols) == count < panel.X.shape[1]
        pred, decision = deployment.fit_predict(name, panel.X, panel.y, panel.Xt, cols, panel.task, seed)
        score, per_unit, ap = deployment.score_predictions(name, panel.gold, pred, panel.units, panel.task, decision)
        quality[method] = score, ap
        cache[f"{method}_cols"], cache[f"{method}_pred"] = cols, pred
        if name == "secom":
            cache[f"{method}_decision"] = decision
        if method in CANDIDATES:
            ordered = name in ("gas","secom")
            assert len(info["folds"]) == (1 if ordered else 2)
            deployment.check_folds(info, panel.groups, ordered, len(panel.y))
            assert np.all(np.isfinite(info["scores"]))
            cache[f"{method}_scores"] = info["scores"]
            for idx, part in enumerate(info["folds"]):
                for key in ("fit_rows", "held_rows", "mate"):
                    cache[f"{method}_{idx}_{key}"] = part[key]
        if method == "full":
            np.testing.assert_array_equal(panel.historical_full, pred)
        units.extend(dict(dataset=name, fold=fold, seed=seed, method=method, unit=u, score=s)
                     for u,s in per_unit.items())
        if verifying is not None:
            audit.append(dict(dataset=name, fold=fold, seed=seed, method=method, width=len(cols),
                              score=score, support_sha256=digest(cols), prediction_sha256=digest(pred)))
    if verifying is not None:
        for key,val in cache.items():
            np.testing.assert_array_equal(val, verifying[key], err_msg=f"{name}-{fold}-{seed}-{key}")
        return audit
    trials = []
    order = np.random.default_rng([SPLIT_SEED, fold, seed]).permutation(METHODS)
    for repeat in range(repeats):
        for pos, method in enumerate(np.roll(order, repeat)):
            start = perf_counter_ns()
            cols, _ = select(method, panel)
            middle = perf_counter_ns()
            pred, decision = deployment.fit_predict(name, panel.X, panel.y, panel.Xt, cols, panel.task, seed)
            end = perf_counter_ns()
            np.testing.assert_array_equal(cache[f"{method}_cols"], cols)
            np.testing.assert_array_equal(cache[f"{method}_pred"], pred)
            if name == "secom":
                np.testing.assert_array_equal(cache[f"{method}_decision"], decision)
            score, ap = quality[method]
            trials.append(dict(dataset=name, fold=fold, seed=seed, method=method, repeat=repeat, position=pos,
                               width=len(cols), p=panel.X.shape[1], score=score, ap=ap,
                               select_ms=(middle-start)/1e6, refit_ms=(end-middle)/1e6, total_ms=(end-start)/1e6))
    return trials, cache, units


def summarize(output, rows, supports):
    cases = five.summarize_timings(output, rows)
    keys = ["dataset","fold","seed"]
    assert cases[keys].drop_duplicates().shape[0] == 69
    assert len(cases) == 69 * len(METHODS)
    earlier = pd.concat([pd.read_csv(f) for f in EARLIER_SCORE_FILES], ignore_index=True)
    earlier = earlier[earlier.method.isin(["full","F_matched","random_matched"])].copy()
    earlier["method"] = earlier.method.replace({"F_matched":"F_75", "random_matched":"random_75"})
    comparison = cases.merge(earlier[keys + ["method","width","score"]], on=keys+["method"],
                             suffixes=("","_previous"), validate="one_to_one")
    assert len(comparison) == 3 * 69
    np.testing.assert_array_equal(comparison.width, comparison.width_previous)
    np.testing.assert_allclose(comparison.score, comparison.score_previous, atol=1e-14, rtol=0)
    comparison[keys+["method","width","score","width_previous","score_previous"]].to_csv(
        output/"prior-control-check.csv", index=False)
    secom = pd.DataFrame(rows).query("dataset == 'secom'").drop_duplicates(["seed","method"])
    secom[["dataset","fold","seed","method","score","ap"]].to_csv(output/"secom-ap.csv", index=False)
    pairs = []
    for (name, fold, method), samples in supports.items():
        for (a,ca),(b,cb) in combinations(samples, 2):
            pairs.append(dict(dataset=name, fold=fold, method=method, a=a, b=b,
                              jaccard=len(np.intersect1d(ca,cb))/len(np.union1d(ca,cb))))
    pd.DataFrame(pairs).to_csv(output/"same-outer-seed-overlap.csv", index=False)


def run(output, repeats, verify):
    current = protocol(repeats)
    manifest = output/"protocol.json"
    if verify:
        assert json.loads(json.dumps(current)) == json.loads(manifest.read_text(encoding="utf-8"))
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(manifest, current)
    rows, units, inputs, audits, supports = [], [], [], [], {}
    for panel in panels():
        name, fold, seed = panel.name, panel.fold, panel.seed
        path = output/f"{name}-{fold}-{seed}.npz"
        inputs.append(dict(dataset=name, fold=fold, seed=seed, task=panel.task, n_train=len(panel.y),
                           n_test=len(panel.gold), p=panel.X.shape[1], train_sha256=digest(panel.X),
                           test_sha256=digest(panel.Xt), y_sha256=digest(panel.y), truth_sha256=digest(panel.gold),
                           groups_sha256="none" if panel.groups is None else digest(panel.groups),
                           units_sha256="none" if panel.units is None else digest(panel.units)))
        if verify:
            with np.load(path) as saved:
                audits.extend(run_panel(panel, repeats, saved))
        else:
            times, values, per_unit = run_panel(panel, repeats)
            rows.extend(times)
            units.extend(per_unit)
            np.savez_compressed(path, **values)
            for method in CANDIDATES:
                supports.setdefault((name,fold,method), []).append((seed,values[f"{method}_cols"]))
        if seed == five.SEEDS[-1] or name not in DATASETS:
            print(f"{'Verified' if verify else 'Measured'} {name} fold {fold} seed {seed}", flush=True)
    assert len(inputs) == 69 and current == protocol(repeats)
    frame = pd.DataFrame(inputs)
    keys = ["dataset","fold","seed"]
    reference = pd.read_csv(DEPLOYMENT_INPUTS).rename(columns={"fit_groups_sha256":"groups_sha256"})
    deployed = frame[frame.dataset.isin(deployment.SOURCES)][reference.columns]
    pd.testing.assert_frame_equal(deployed.sort_values(keys).reset_index(drop=True),
                                  reference.sort_values(keys).reset_index(drop=True), check_dtype=False)
    if verify:
        pd.testing.assert_frame_equal(frame, pd.read_csv(output/"inputs.csv"), check_dtype=False)
        pd.DataFrame(audits).to_csv(output/"fresh-process-audit.csv", index=False)
    else:
        frame.to_csv(output/"inputs.csv", index=False)
        pd.DataFrame(units).to_csv(output/"unit_scores.csv", index=False)
        summarize(output, rows, supports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("Use at least five timing repeats")
    with threadpool_limits(limits=1):
        run(args.output, args.repeats, args.verify)
