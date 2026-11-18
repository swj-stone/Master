"""Frozen transfer panels for two budget-adaptive minPI rules and width-matched baselines."""
import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from statsmodels.datasets import macrodata
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.fast_minpi_five_source import check_columns, summarize_timings
from experiments.minpi_budget_transfer import SEEDS as OLD_SEEDS, panels as old_panels
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric, prediction
from minppi_mofs.pilot_minpi import pilot_top_from_training
from minppi_mofs.sketch_minpi import reduced_width
from minppi_mofs.tournament_minpi import tournament_top_from_training


SPLITS = (39791, 41809, 57139)
MECHANISMS = ("proxy", "sparse", "dense")
TASKS = ("classification", "regression")
BASE = ("tournament", "pilot", "random_tournament", "random_pilot", "F_tournament", "F_pilot", "full")
FIXED = ("tournament_75", "pilot_75", "random_75", "F_75")
SELECTORS = {"tournament": tournament_top_from_training, "pilot": pilot_top_from_training}
OLD = ROOT / "results/minpi-study/moment-transfer-20261117a"


@dataclass
class Panel:
    name: str
    family: str
    mechanism: str
    task: str
    fold: int
    seed: int
    X: np.ndarray
    y: np.ndarray
    Xt: np.ndarray
    truth: np.ndarray
    split_info: dict
    groups: np.ndarray | None = None
    ordered: bool = False


def redundant_table(X):
    """Training-only evaluation stratum fixed before benchmarking; never passed to selectors."""
    n, p = X.shape
    if n < 4 * p or not 16 <= p <= 512:
        return False, np.nan
    sample = X[:256].astype(np.float64)
    sample -= sample.mean(axis=0)
    sample /= np.maximum(np.linalg.norm(sample, axis=0), np.finfo(float).tiny)
    corr = np.abs(sample.T @ sample)
    np.fill_diagonal(corr, 0.)
    fraction = (corr.max(axis=0) >= .9).mean()
    return bool(fraction >= .25), float(fraction)


def generated(mechanism, task, p, replicate):
    seed = [20261117, MECHANISMS.index(mechanism), TASKS.index(task), p, replicate]
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2400, p))
    signal_width = p if mechanism == "dense" else 8
    signs = rng.choice([-1., 1.], signal_width) / np.sqrt(signal_width)
    signal = X[:, :signal_width] @ signs
    if mechanism == "proxy":
        # The first half of columns measure eight latent informative variables.
        for j in range(8, p // 2):
            X[:, j] = X[:, j % 8] + .1 * X[:, j]
    target = signal + .3 * rng.normal(size=len(X))
    if task == "classification":
        target = (target > 0).astype(int)
    X = np.ascontiguousarray(X[:, rng.permutation(p)], dtype=np.float32)
    return Panel(f"{mechanism}_{task}_{p}", "new_simulation", mechanism, task, replicate,
                 OLD_SEEDS[0], X[:1600], target[:1600], X[1600:], target[1600:],
                 {"generation_seed": seed, "n_train": 1600})


def panels():
    for name, X, y, Xt, truth, task in old_panels():
        family = "previous_real" if name in ("arcene", "dexter") else "previous_simulation"
        for seed in OLD_SEEDS:
            yield Panel(name, family, name, task, 1, seed, X, y, Xt, truth,
                        {"historical_trace": f"{name}.npz"})
    for name, loader in (("breast_cancer", load_breast_cancer), ("wine", load_wine)):
        X, y = loader(return_X_y=True)
        for fold, split_seed in enumerate(SPLITS, 1):
            tr, te = train_test_split(np.arange(len(y)), test_size=1/3,
                                      random_state=split_seed, stratify=y)
            yield Panel(name, "new_real", name, "classification", fold, OLD_SEEDS[fold-1],
                        X[tr], y[tr], X[te], y[te],
                        {"split_seed": split_seed, "train_rows": tr.tolist(), "test_rows": te.tolist()})
    for mechanism in MECHANISMS:
        for task in TASKS:
            for p in (48, 128):
                for replicate in (1, 2, 3):
                    yield generated(mechanism, task, p, replicate)
    # Secondary real replication announced after defining the 54 primary cases,
    # before generating candidate outcomes. Levels are the archived vintage.
    source = macrodata.load_pandas().data
    levels = source.drop(columns=["year", "quarter"]).to_numpy()
    features = np.ascontiguousarray(np.column_stack((levels[1:-1], levels[:-2])))
    target = np.diff(np.log(source.realgdp.to_numpy()))[1:]
    years = source.year.to_numpy()[1:-1].astype(int)
    for fold, ntrain in enumerate((131, 151, 171), 1):
        # Embargo one row so every training target predates the first test row.
        test = np.arange(ntrain + 1, ntrain + 21)
        yield Panel("us_macro_gdp", "additional_real", "lagged_measurements", "regression",
                    fold, OLD_SEEDS[fold-1], features[:ntrain], target[:ntrain],
                    features[test], target[test],
                    {"train_rows": [0, ntrain-1], "test_rows": test.tolist(), "embargo_row": ntrain},
                    years[:ntrain], True)


def methods(panel):
    return BASE + FIXED if panel.family.startswith("previous") else BASE


def select(method, panel, widths):
    X, y = panel.X, panel.y
    p = X.shape[1]
    if method == "full":
        return np.arange(p)
    name, _, suffix = method.partition("_")
    if name in SELECTORS:
        count = reduced_width(p, .75) if suffix == "75" else None
        return SELECTORS[name](X, y, count, task=panel.task, random_state=panel.seed,
                               groups=panel.groups, ordered_groups=panel.ordered)[0]
    count = reduced_width(p, .75) if suffix == "75" else widths[suffix]
    return choose(name.lower(), X, y, panel.task, panel.seed, count)


def protocol():
    files = ("experiments/adaptive_minpi_transfer.py", "experiments/minpi_budget_transfer.py",
             "experiments/minpi_study_external.py", "experiments/stable_minpi_stress.py",
             "experiments/fast_minpi_five_source.py", "experiments/compare_predictive_minpi.py",
             "experiments/minpi_study.py", "configs/minpi-study.json",
             "src/minppi_mofs/pilot_minpi.py", "src/minppi_mofs/tournament_minpi.py",
             "src/minppi_mofs/moment_minpi.py", "src/minppi_mofs/sketch_minpi.py", "src/minppi_mofs/utility_minpi.py")
    return dict(hashes={f: file_hash(ROOT / f) for f in files},
                old_hashes={p.name: file_hash(p) for p in sorted(OLD.glob("*.npz"))},
                secondary_source_sha256=file_hash(Path(macrodata.__file__).with_name("macrodata.csv")),
                old_seeds=list(OLD_SEEDS),
                new_real_split_seeds=list(SPLITS), new_real_selector_seeds=list(OLD_SEEDS),
                simulation_selector_seed=OLD_SEEDS[0], simulation_mechanisms=list(MECHANISMS),
                n_simulated_instances=36, old_case_count=12, new_real_case_count=6,
                n_distinct_new_real_sources=2, secondary_real="US quarterly macrodata, three chronological "
                "20-quarter test periods, one-row pre-test embargo. Single archived vintage, no vintage "
                "release-date reconstruction.", repeats=5, threads=1, methods=list(BASE),
                previous_panels_extra_methods=list(FIXED), scipy=scipy.__version__,
                sklearn=sklearn.__version__, numpy=np.__version__, platform=platform.platform(),
                python=platform.python_version(),
                target_class="n>=4p, 16<=p<=512, >=25% of features have a |r|>=.9 partner "
                             "in the first up to 256 training rows (constant correlations set to zero).",
                metric="balanced accuracy / R-squared. Strict width<p, gap>=-.02, "
                       "median(total)<median(full). Secondary 5% and 10% time margins.",
                comparison="One full warmup then five fresh, single-threaded rotated fits with the "
                           "same standardized ridge on all outer training rows. Candidate timing "
                           "includes width choice. Matched F and random receive each candidate's "
                           "training-only chosen count for free. All column slicing is timed. "
                           "Secondary routed cost adds one training-table diagnostic to each repeat "
                           "of the candidate. The full and matched controls receive no routing charge.",
                interpretation="Compare the prespecified observable class against its complement, "
                               "the three known signal mechanisms, and each real source separately. "
                               "Seeds on earlier panels share test samples, new real splits may overlap. "
                               "A descriptive specialization is supported only if on the qualifying "
                               "new real panels and on qualifying new simulated classification and "
                               "regression panels separately a candidate gets a majority of joint "
                               "successes and meets/exceeds each matched F/random count.")


def run_case(panel, order_seed, repeats, trace, verify, arms, chooser, adaptive_names):
    widths, scores, arrays, rows, audit = {}, {}, {}, [], []
    arrays["input_hashes"] = np.asarray([digest(a) for a in (panel.X, panel.y, panel.Xt, panel.truth)])
    arrays["truth"] = panel.truth
    if panel.groups is not None:
        arrays["groups"] = panel.groups
    for method in arms:
        cols = chooser(method, panel, widths)
        check_columns(cols, panel.X.shape[1])
        widths[method] = len(cols)
        if method in adaptive_names:
            widths[adaptive_names[method]] = len(cols)
        pred = prediction(panel.X, panel.y, panel.Xt, cols, panel.task, panel.seed)
        scores[method] = metric(panel.truth, pred, panel.task)
        arrays[f"{method}_cols"] = cols
        arrays[f"{method}_pred"] = pred
    for candidate, budget in adaptive_names.items():
        assert widths[candidate] == widths["F_" + budget] == widths["random_" + budget]
    if "F_75" in arms:
        with np.load(OLD / (panel.name + ".npz")) as source:
            np.testing.assert_array_equal(arrays["input_hashes"], source["input_hashes"])
            for method in ("full", "F_75", "random_75"):
                for key in ("cols", "pred"):
                    np.testing.assert_array_equal(arrays[f"{method}_{key}"],
                                                  source[f"{method}_{panel.seed}_{key}"])
        assert all(widths[name] == reduced_width(panel.X.shape[1], .75) for name in arms if name.endswith("_75"))
    if verify:
        with np.load(trace) as original:
            assert set(arrays) == set(original.files)
            for key in arrays:
                np.testing.assert_array_equal(arrays[key], original[key])
        for method in arms:
            audit.append(dict(dataset=panel.name, fold=panel.fold, seed=panel.seed, method=method,
                              support_sha256=digest(arrays[f"{method}_cols"]),
                              prediction_sha256=digest(arrays[f"{method}_pred"])))
        return rows, audit
    np.savez_compressed(trace, **arrays)
    order = np.random.default_rng(order_seed).permutation(arms)
    for r in range(repeats):
        start_diag = perf_counter_ns()
        redundant_table(panel.X)
        diagnostic_ms = (perf_counter_ns() - start_diag) / 1e6
        for position, method in enumerate(np.roll(order, r)):
            start = perf_counter_ns()
            cols = chooser(method, panel, widths)
            chosen = perf_counter_ns()
            pred = prediction(panel.X, panel.y, panel.Xt, cols, panel.task, panel.seed)
            finish = perf_counter_ns()
            np.testing.assert_array_equal(cols, arrays[f"{method}_cols"])
            np.testing.assert_array_equal(pred, arrays[f"{method}_pred"])
            rows.append(dict(dataset=panel.name, fold=panel.fold, seed=panel.seed, method=method,
                             repeat=r, position=position, width=len(cols), p=panel.X.shape[1],
                             score=scores[method], diagnostic_ms=diagnostic_ms, select_ms=(chosen-start)/1e6,
                             refit_ms=(finish-chosen)/1e6, total_ms=(finish-start)/1e6))
    return rows, audit


def summaries(out, rows, metadata, expected, adaptive_names):
    cases = summarize_timings(out, rows)
    frame = pd.DataFrame(metadata)
    assert len(frame) == 57 and len(cases) == expected
    frame.to_csv(out/"panels.csv", index=False)
    cases = cases.merge(frame, on=["dataset", "fold", "seed"], validate="many_to_one")
    cost = pd.DataFrame(rows)
    cost["routed_total_ms"] = cost.total_ms + cost.diagnostic_ms * cost.method.isin(adaptive_names)
    keys = ["dataset", "fold", "seed", "method"]
    total = cost.groupby(keys, as_index=False).agg(diagnostic_ms=("diagnostic_ms", "median"),
                                                   routed_total_ms=("routed_total_ms", "median"))
    cases = cases.merge(total, on=keys, validate="one_to_one")
    cases["routed_joint"] = cases.reduced_pass & (cases.routed_total_ms < cases.full_total_ms)
    cases.to_csv(out/"cases-with-routing.csv", index=False)
    cases["retained"] = cases.width / cases.p
    cases["joint_no_loss"] = cases.joint & (cases.gap >= 0)
    spec = dict(configs=("dataset", "size"), sources=("dataset", "nunique"),
                reduced=("reduced", "sum"), quality=("within", "sum"),
                joint=("joint", "sum"), routed_joint=("routed_joint", "sum"),
                joint_5pct=("joint_5pct", "sum"), joint_10pct=("joint_10pct", "sum"),
                joint_no_loss=("joint_no_loss", "sum"), median_fraction=("retained", "median"),
                median_time_ratio=("ratio", "median"), median_select_ms=("select_ms", "median"),
                worst_gap=("gap", "min"), mean_gap=("gap", "mean"))
    for keys, filename in ((["family", "observable_class"], "class-by-family"),
                           (["observable_class"], "observable-class"),
                           (["family", "observable_class", "task"], "class-by-task"),
                           (["family", "mechanism", "task"], "mechanism")):
        cases.groupby([*keys, "method"]).agg(**spec).to_csv(out/f"{filename}.csv")


def run(out, verify, *, make_protocol=protocol, get_methods=methods, chooser=select, adaptive_names=None):
    if adaptive_names is None:
        adaptive_names = {name: name for name in SELECTORS}
    state = make_protocol()
    manifest = out / "protocol.json"
    if verify:
        assert json.loads(manifest.read_text(encoding="utf-8")) == state
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest.write_text(json.dumps(state, indent=2), encoding="utf-8")
    rows, audit, metadata, splits = [], [], [], {}
    expected = 0
    for idx, panel in enumerate(panels()):
        stem = f"{panel.name}-{panel.fold}-{panel.seed}"
        splits[stem] = panel.split_info
        qualifying, share = redundant_table(panel.X)
        metadata.append(dict(dataset=panel.name, fold=panel.fold, seed=panel.seed, family=panel.family,
                             mechanism=panel.mechanism, task=panel.task, n_train=len(panel.X),
                             n_test=len(panel.Xt), observable_class=qualifying, correlated_share=share))
        arms = get_methods(panel)
        expected += len(arms)
        records, entries = run_case(panel, [88471, idx], state["repeats"], out/(stem+".npz"), verify,
                                    arms, chooser, adaptive_names)
        rows.extend(records)
        audit.extend(entries)
        print("verified" if verify else "measured", stem, flush=True)
    assert make_protocol() == state
    if verify:
        assert json.loads((out/"splits.json").read_text(encoding="utf-8")) == splits
        assert len(splits) == 57 and len(audit) == expected
        pd.DataFrame(audit).to_csv(out/"fresh-process-audit.csv", index=False)
    else:
        (out/"splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
        summaries(out, rows, metadata, expected, adaptive_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
