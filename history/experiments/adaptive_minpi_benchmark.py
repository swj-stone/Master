"""Replay all 69 archived configurations with training-selected widths and matched controls."""
import argparse
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import budget_minpi_deployments as deployment
from experiments import fast_minpi_five_source as five
from experiments import moment_minpi_benchmark as previous
from experiments.compare_predictive_minpi import choose
from experiments.stable_minpi_stress import ROOT, SPLIT_SEED, digest, file_hash, write_json
from minppi_mofs.pilot_minpi import pilot_top_from_training
from minppi_mofs.sketch_minpi import reduced_width
from minppi_mofs.stable_minpi import _impute_validation
from minppi_mofs.tournament_minpi import tournament_top_from_training


RUN = deployment.RUNS / "moment-69-20261117a"
SELECTORS = {"pilot": pilot_top_from_training, "tournament": tournament_top_from_training}
CANDIDATES = ("pilot_auto", "tournament_auto", "pilot_75", "tournament_75")
MATCHED = ("random_pilot", "F_pilot", "random_tournament", "F_tournament")
ARCHIVED = ("full", "moment_75", "random_75", "F_75")
METHODS = CANDIDATES + MATCHED + ARCHIVED
CASE = ["dataset", "fold", "seed"]


def select(method, panel, widths):
    if method in ARCHIVED:
        return previous.select(method, panel)
    name, mode = method.split("_")
    if name in SELECTORS:
        count = None if mode == "auto" else reduced_width(panel.X.shape[1], .75)
        return SELECTORS[name](panel.X, panel.y, count, task=panel.task, random_state=panel.seed,
                               groups=panel.groups, ordered_groups=panel.name in ("gas", "secom"))
    # The baseline receives the candidate width for free. Every timed adaptive
    # minPI call recomputes its width and the indices from scratch.
    width = widths[mode]
    X = panel.X
    if name == "F" and panel.name in deployment.ADDITIONAL:
        X, _ = _impute_validation(X, X[:0])
    return choose(name.lower(), X, panel.y, panel.task, panel.seed, width), {}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported diagnostic {type(value)}")


def run_panel(panel, repeats, reference=None):
    """Return timings, exact outputs, scalar diagnostics, unit scores and replay checks."""
    name, fold, seed = panel.name, panel.fold, panel.seed
    key = dict(dataset=name, fold=fold, seed=seed)
    stored, widths, scores, diagnostics, units, audits = {}, {}, {}, [], [], []
    p = panel.X.shape[1]
    for method in METHODS:
        columns, info = select(method, panel, widths)
        five.check_columns(columns, p)
        if method.endswith("_auto"):
            widths[method.split("_")[0]] = len(columns)
        elif method in MATCHED:
            assert len(columns) == widths[method.split("_")[1]]
        elif method.endswith("_75"):
            assert len(columns) == reduced_width(p, .75)
        if method in CANDIDATES:
            splits = info.get("folds", [])
            if splits and all(isinstance(part, dict) and "fit_rows" in part and "held_rows" in part
                              for part in splits):
                deployment.check_folds(info, panel.groups, name in ("gas", "secom"), len(panel.X))
            record = json.dumps(info, default=json_value, sort_keys=True, separators=(",", ":"))
            stored[f"{method}_diagnostics"] = np.asarray(record)
            scalars = {f"diag_{k}": v for k, v in info.items() if np.isscalar(v)}
            diagnostics.append(dict(**key, method=method, p=p, width=len(columns), **scalars))
        pred, decision = deployment.fit_predict(name, panel.X, panel.y, panel.Xt, columns, panel.task, seed)
        score, unit_scores, ap = deployment.score_predictions(name, panel.gold, pred, panel.units,
                                                              panel.task, decision)
        stored[f"{method}_cols"] = columns
        stored[f"{method}_pred"] = pred
        if name == "secom":
            stored[f"{method}_decision"] = decision
        scores[method] = (score, ap)
        units.extend(dict(**key, method=method, unit=unit, score=value) for unit, value in unit_scores.items())
        if method == "full" and panel.historical_full is not None:
            np.testing.assert_array_equal(pred, panel.historical_full)
        if reference is not None:
            audits.append(dict(**key, method=method, width=len(columns), score=score,
                               support_sha256=digest(columns), prediction_sha256=digest(pred)))
    if reference is not None:
        assert set(stored) == set(reference.files)
        for field, array in stored.items():
            np.testing.assert_array_equal(array, reference[field], err_msg=field)
        return [], stored, diagnostics, units, audits

    records = []
    order = np.random.default_rng([SPLIT_SEED, fold, seed]).permutation(METHODS).tolist()
    for repeat in range(repeats):
        shift = (repeat * 5) % len(order)
        for pos, method in enumerate(order[shift:] + order[:shift]):
            start = perf_counter_ns()
            cols, _ = select(method, panel, widths)
            middle = perf_counter_ns()
            pred, decision = deployment.fit_predict(name, panel.X, panel.y, panel.Xt, cols, panel.task, seed)
            end = perf_counter_ns()
            np.testing.assert_array_equal(cols, stored[f"{method}_cols"])
            np.testing.assert_array_equal(pred, stored[f"{method}_pred"])
            if name == "secom":
                np.testing.assert_array_equal(decision, stored[f"{method}_decision"])
            score, ap = scores[method]
            records.append(dict(**key, method=method, repeat=repeat, position=pos, width=len(cols), p=p,
                                score=score, ap=ap, select_ms=(middle-start)/1e6,
                                refit_ms=(end-middle)/1e6, total_ms=(end-start)/1e6))
    return records, stored, diagnostics, units, audits


def protocol(repeats):
    current = previous.protocol(repeats)
    for p in (Path(__file__), ROOT/"src/minppi_mofs/tournament_minpi.py",
              ROOT/"src/minppi_mofs/pilot_minpi.py"):
        current["source_sha256"][p.relative_to(ROOT).as_posix()] = file_hash(p)
    for p in (RUN/"inputs.csv", RUN/"cases.csv", RUN/"protocol.json", *sorted(RUN.glob("*.npz"))):
        current["archives"][p.relative_to(ROOT).as_posix()] = file_hash(p)
    current.update(methods=list(METHODS), primary="Two prespecified automatic widths, each with its own controls",
                   sensitivity="75% clamp for both scorers and the archived moment rule with fixed controls",
                   budgets="Auto from training data; the fixed width is reduced_width(p, .75)",
                   controls="ANOVA/classification or univariate regression F; uniform sampling without "
                            "replacement using the archived selector seed and the identical actual width",
                   timing="Warm every method. Rotate a seeded permutation by 5 of 12 positions each round. "
                          "Each round recomputes selection (including automatic width), column slicing, "
                          "historical training-only imputation, standardization, fitting and prediction. "
                          "Baseline width is supplied without charging the baseline for learning it.",
                   groups="Source and split IDs go only to the experimental loader. Selectors receive "
                          "training X/y, seed, group labels and the time-order flag.")
    return current


def input_hashes(panel):
    return dict(dataset=panel.name, fold=panel.fold, seed=panel.seed, task=panel.task,
                n_train=len(panel.y), n_test=len(panel.gold), p=panel.X.shape[1],
                train_sha256=digest(panel.X), test_sha256=digest(panel.Xt), y_sha256=digest(panel.y),
                truth_sha256=digest(panel.gold),
                groups_sha256="none" if panel.groups is None else digest(panel.groups),
                units_sha256="none" if panel.units is None else digest(panel.units))


def score_controls(output, cases):
    old = pd.read_csv(RUN/"cases.csv")
    saved = old[old.method.isin(ARCHIVED)][CASE + ["method", "width", "score", "joint"]]
    rematch = cases.merge(saved, on=CASE+["method"], suffixes=("", "_prior"), validate="one_to_one")
    assert len(rematch) == len(ARCHIVED) * 69
    np.testing.assert_array_equal(rematch.width, rematch.width_prior)
    np.testing.assert_allclose(rematch.score, rematch.score_prior, rtol=0, atol=1e-14)
    rematch.to_csv(output/"archived-controls.csv", index=False)


def summarize(output, trials, supports):
    cases = five.summarize_timings(output, trials)
    assert len(cases) == len(METHODS)*69 and len(cases[CASE].drop_duplicates()) == 69
    score_controls(output, cases)
    trials = pd.DataFrame(trials)
    secom = trials[trials.dataset == "secom"].drop_duplicates(["seed", "method"])
    secom[["dataset","fold","seed","method","score","ap"]].to_csv(output/"secom-ap.csv", index=False)
    splits = []
    for (name, fold, method), selections in supports.items():
        for (a, ca), (b, cb) in combinations(selections, 2):
            splits.append(dict(dataset=name, fold=fold, method=method, a=a, b=b,
                               jaccard=len(np.intersect1d(ca, cb))/len(np.union1d(ca, cb))))
    pd.DataFrame(splits).to_csv(output/"same-outer-seed-overlap.csv", index=False)
    # Sensitivity to measurement jitter: score and width are fixed in every repeat.
    for label, subset in (("first", trials[trials.repeat < 3]),
                          ("last", trials[trials.repeat >= trials.repeat.max() - 2])):
        times = subset.groupby(CASE+["method"], as_index=False).total_ms.median()
        full = times[times.method == "full"][CASE+["total_ms"]].rename(columns={"total_ms":"full_ms"})
        times = times.merge(full, on=CASE, validate="many_to_one")
        times[label] = times.total_ms < times.full_ms
        cases = cases.merge(times[CASE+["method",label]], on=CASE+["method"], validate="one_to_one")
    cases["joint_both_halves"] = cases.joint & cases["first"] & cases["last"]
    cases["joint_no_score_loss"] = cases.joint & (cases.gap >= 0)
    cases["keep_fraction"] = cases.width / cases.p
    stats = dict(cases=("fold","size"), mean_keep=("keep_fraction","mean"),
                 median_select_ms=("select_ms","median"), median_ratio=("ratio","median"),
                 within=("within","sum"), joint=("joint","sum"), joint_5pct=("joint_5pct","sum"),
                 joint_both_halves=("joint_both_halves","sum"), joint_no_loss=("joint_no_score_loss","sum"))
    cases.groupby("method").agg(**stats).to_csv(output/"sensitivity.csv")
    cases.groupby(["dataset","method"]).agg(**stats).to_csv(output/"source-sensitivity.csv")
    # Pair successful runs at identical budgets to distinguish selector cost
    # from the ability of the selected support to preserve the outer score.
    comparisons = []
    for method in CANDIDATES:
        name, mode = method.split("_")
        left = cases[cases.method == method]
        suffix = name if mode == "auto" else "75"
        for base in ("random", "F"):
            right = cases[cases.method == f"{base}_{suffix}"]
            paired = left.merge(right, on=CASE, suffixes=("_minpi", "_base"), validate="one_to_one")
            np.testing.assert_array_equal(paired.width_minpi, paired.width_base)
            paired["only_minpi"] = paired.joint_minpi & ~paired.joint_base
            paired["only_control"] = paired.joint_base & ~paired.joint_minpi
            paired["both"] = paired.joint_base & paired.joint_minpi
            paired["control_only_minpi_quality_passed"] = paired.only_control & paired.within_minpi
            paired["minpi_score_gain"] = paired.score_minpi - paired.score_base
            paired["selector_overhead_ms"] = paired.select_ms_minpi - paired.select_ms_base
            comparisons.append(paired)
    pd.concat(comparisons, ignore_index=True).to_csv(output/"matched-comparisons.csv", index=False)


def run(output, repeats, verify):
    current = protocol(repeats)
    path = output/"protocol.json"
    if verify:
        assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(current))
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(path, current)
    records, inputs, diagnostics, unit_scores, audited, supports = [], [], [], [], [], {}
    for panel in previous.panels():
        name, fold, seed = panel.name, panel.fold, panel.seed
        filename = f"{name}-{fold}-{seed}.npz"
        inputs.append(input_hashes(panel))
        if verify:
            with np.load(output/filename) as baseline:
                _, _, _, _, audits = run_panel(panel, repeats, baseline)
                audited.extend(audits)
        else:
            rows, values, log, scores, _ = run_panel(panel, repeats)
            with np.load(RUN/filename) as baseline:
                for method in ARCHIVED:
                    for suffix in ("cols", "pred", *(("decision",) if name == "secom" else ())):
                        field = f"{method}_{suffix}"
                        np.testing.assert_array_equal(values[field], baseline[field], err_msg=field)
            np.savez_compressed(output/filename, **values)
            records.extend(rows)
            unit_scores.extend(scores)
            diagnostics.extend(log)
            for method in CANDIDATES:
                supports.setdefault((name, fold, method), []).append((seed, values[f"{method}_cols"]))
        if seed == five.SEEDS[-1] or name not in previous.DATASETS:
            print(f"{'Verified' if verify else 'Measured'} {name} fold {fold} seed {seed}", flush=True)
    assert current == protocol(repeats)
    frame = pd.DataFrame(inputs)
    assert len(frame) == 69 and not frame.duplicated(CASE).any()
    pd.testing.assert_frame_equal(frame, pd.read_csv(RUN/"inputs.csv"), check_dtype=False)
    if verify:
        pd.testing.assert_frame_equal(frame, pd.read_csv(output/"inputs.csv"), check_dtype=False)
        pd.DataFrame(audited).to_csv(output/"fresh-process-audit.csv", index=False)
    else:
        frame.to_csv(output/"inputs.csv", index=False)
        pd.DataFrame(unit_scores).to_csv(output/"unit-scores.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(output/"diagnostics.csv", index=False)
        summarize(output, records, supports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.repeats < 7:
        parser.error("At least seven repeats are required")
    with threadpool_limits(limits=1):
        run(args.output, args.repeats, args.verify)
