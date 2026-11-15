"""Compare nomination, guard and complete elapsed times on the original XGBoost slates."""
import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import transport_predict as primary
from experiments.context_review.decision_opportunities import EVIDENCE, PHASES, RUNS
from experiments.context_review.low_fidelity_control import minimum_slack, small_scores
from experiments.context_review.matched_stopping import choose, context_score
from experiments.context_review.timing_process import bind_logical_cpu

CONFIG = bg.ROOT / "configs/context_review/budget_comparison.json"
DEST = EVIDENCE / "budget_comparison"
METHODS = ("point", "slack", "small")
KEY = ["panel", "dataset", "fold", "method"]


def signature(probs):
    return sha256(np.ascontiguousarray(probs).tobytes()).hexdigest()


def measure(data, dataset_id, fold, method):
    x, y, outer, train, prop, guard = (data[k] for k in ("x", "y", "x_outer", "train", "proposal", "guard"))
    seed = 20421 + fold
    start = perf_counter()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    xt, xp = imputer.transform(x[train]), imputer.transform(x[prop])
    model = bg.fit(xt, y[train], seed)
    binary = model.n_classes_ == 2
    rare = bool(binary and min(np.bincount(y[prop])) <= .05 * len(prop))
    names = [f"c{i}" for i in range(x.shape[1])]
    _, info = bg.gain_anchor(model, pd.DataFrame(xt, columns=names), y[train], pd.DataFrame(xp, columns=names),
                             y[prop], task="binary" if binary else "multiclass", utility="ovr_auc",
                             seed=seed, aggregate="gain")
    order = [int(s[1:]) for s in info["order"]]
    k = int(info["k"])
    anchor = tuple(sorted(order[:k]))
    width = (k + 1) // 2
    candidates = bg.candidate_pool(order[:k], width, 1000 + dataset_id, fold)[1] if width < k else []
    supports = [anchor, *candidates]
    nomination_start = perf_counter()
    index = 0
    masks = small_fits = 0
    if candidates and method == "small":
        scores, _ = small_scores(x, y, train, prop, supports, seed, rare)
        small_fits = len(scores)
        index = max(range(1, len(scores)), key=lambda i: minimum_slack(scores[i], scores[0]))
    elif candidates:
        point = method == "point"
        ballot = primary.build_ballot(model, xt, xp, y[prop], seed, anchor, rare, point=point)
        family = (frozenset(),) if point else primary.separating_contexts(order[k:])

        def evaluate(i, restored):
            after = ballot.score(frozenset(supports[i]) | restored)
            before = ballot.score(ballot.anchor | restored)
            return context_score(after, before, "point" if point else "margin")

        index = choose(range(1, len(supports)), family, evaluate, "point" if point else "margin", not point)
        masks = len(ballot.scores)
    guard_start = perf_counter()
    final = 0
    if index:
        checked, _, _ = bg.predict_sets(x, y, np.sort(np.r_[train, prop]), x[guard],
                                         [anchor, supports[index]], seed)
        passes = bg._within(bg.metrics(y[guard], checked[supports[index]], rare),
                            bg.metrics(y[guard], checked[anchor], rare))
        final = index if passes else 0
    fit_start = perf_counter()
    delivered = supports[final]
    probs, _, _ = bg.predict_sets(x, y, np.arange(len(x)), outer, [delivered], seed)
    end = perf_counter()
    return supports, dict(nominee=index, delivered=final, small_fits=small_fits, masks=masks,
                          nomination_s=guard_start - nomination_start, setup_s=nomination_start - start,
                          guard_s=fit_start - guard_start, final_s=end - fit_start, total_s=end - start,
                          probability_sha256=signature(probs[delivered]))


def validate_row(row, selection, sealed, probabilities):
    supports = sealed["supports"]
    i = sealed["low_fidelity"] if row["method"] == "small" else supports.index(selection["choices"][row["method"]]["nominee"])
    j = i if bg._within(sealed["guard_metrics"][i], sealed["guard_metrics"][0]) else 0
    assert (row["nominee"], row["delivered"]) == (i, j)
    assert row["probability_sha256"] == signature(probabilities[f"o{j}"])
    assert row["small_fits"] == (sealed["fits_8"] if row["method"] == "small" else 0)
    assert row["masks"] >= 0 and (row["masks"] > 0) == (i != 0 and row["method"] != "small")
    stages = [row[k] for k in ("nomination_s", "setup_s", "guard_s", "final_s")]
    assert all(s >= 0 for s in stages) and np.isclose(sum(stages), row["total_s"], rtol=0, atol=1e-9)


def jobs():
    for panel, (source, slate, seal_file) in PHASES.items():
        for key, receipt in sorted(bg.read(EVIDENCE / seal_file).items()):
            folder = RUNS / slate / key
            sealed = bg.read(folder / "sealed.json")
            assert bg.digest(folder / "sealed.json") == receipt
            assert bg.digest(folder / "predictions.npz") == sealed["predictions"]
            yield panel, key, RUNS / source / key, folder, sealed


def timing():
    from experiments.context_review.matched_stopping_audit import environment
    config = bg.read(CONFIG)
    assert tuple(config["methods"]) == METHODS and config["repetitions"] == len(METHODS)
    DEST.mkdir(parents=True, exist_ok=True)
    cpu = bind_logical_cpu()
    stamp = dict(source=bg.digest(Path(__file__)), protocol=bg.digest(CONFIG), cpu=cpu)
    records = []
    begun = perf_counter()
    with threadpool_limits(limits=1):
        for number, (panel, key, src, folder, sealed) in enumerate(jobs(), 1):
            layout = bg.read(src / "layout.json")
            selection = bg.read(src / "selection.json")
            checkpoint = RUNS / "budget-comparison" / stamp["source"][:12] / panel / key / "timing.json"
            if checkpoint.exists():
                saved = bg.read(checkpoint)
                assert saved["stamp"] == stamp and len(saved["rows"]) == 9
                records.extend(saved["rows"])
            else:
                assert bg.digest(src / "inputs.npz") == layout["inputs_sha256"]
                with np.load(src / "inputs.npz") as raw:
                    data = {k: raw[k] for k in raw.files}
                rng = np.random.default_rng([config["order_seed"], layout["id"], layout["fold"]])
                order = rng.permutation(METHODS).tolist()
                rows = []
                with np.load(folder / "predictions.npz") as probs:
                    for rep in range(config["repetitions"]):
                        for position, method in enumerate(order[rep:] + order[:rep]):
                            supports, result = measure(data, layout["id"], layout["fold"], method)
                            assert list(map(list, supports)) == sealed["supports"]
                            row = dict(panel=panel, dataset=layout["dataset"], fold=layout["fold"], method=method,
                                       repeat=rep, position=position, **result)
                            validate_row(row, selection, sealed, probs)
                            rows.append(row)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                stage = checkpoint.with_suffix(".partial")
                bg.store(stage, dict(stamp=stamp, rows=rows))
                stage.replace(checkpoint)
                records.extend(rows)
            elapsed = perf_counter() - begun
            print(f"[{number}/46] {panel}/{key} elapsed {elapsed:.1f}s ETA {elapsed*(46-number)/number:.1f}s", flush=True)
    pd.DataFrame(records).to_csv(DEST / "timings.csv", index=False)
    machine = environment()
    machine.pop("forest_n_jobs")
    machine.update(xgboost_n_jobs=1, cpu_affinity=f"logical CPU {cpu}" if cpu is not None else "not pinned")
    (DEST / "timed-source.py").write_bytes(Path(__file__).read_bytes())
    bg.store(DEST / "receipt.json", dict(**stamp, environment=machine, pinned_logical_cpu=cpu,
                                         timings_sha256=bg.digest(DEST / "timings.csv")))
    replay()


def replay():
    receipt = bg.read(DEST / "receipt.json")
    assert bg.digest(DEST / "timed-source.py") == receipt["source"]
    assert bg.digest(DEST / "timings.csv") == receipt["timings_sha256"]
    data = pd.read_csv(DEST / "timings.csv", float_precision="round_trip")
    assert len(data) == 46 * 3 * 3 and not data.duplicated(KEY + ["repeat"]).any()
    means = []
    for panel, key, src, folder, sealed in jobs():
        task, fold = sealed["dataset"], sealed["fold"]
        rows = data[(data.panel == panel) & (data.dataset == task) & (data.fold == fold)]
        selection = bg.read(src / "selection.json")
        with np.load(folder / "predictions.npz") as probs:
            for row in rows.to_dict("records"):
                validate_row(row, selection, sealed, probs)
        for method in METHODS:
            r = rows[rows.method == method]
            assert sorted(r.position) == sorted(r.repeat) == [0, 1, 2]
            d = r.iloc[0]
            means.append(dict(panel=panel, dataset=task, fold=fold, method=method,
                              nominee=d.nominee, delivered=d.delivered, masks=d.masks, small_fits=d.small_fits,
                              nomination_s=r.nomination_s.median(), total_s=r.total_s.median()))
    costs = pd.DataFrame(means)
    outcomes = []
    for panel in PHASES:
        y = pd.read_csv(EVIDENCE / "chance" / f"{panel}-folds.csv")
        outcomes.append(y[(y.tolerance == .03) & y.method.isin(METHODS)].assign(panel=panel))
    full = costs.merge(pd.concat(outcomes)[KEY + ["u", "breach", "auc", "accuracy"]], on=KEY, validate="one_to_one")
    assert len(full) == 46 * 3
    fields = ["u", "breach", "auc", "accuracy"]
    by_task = full.groupby(["panel", "dataset", "method"])[fields].mean()
    by = ["panel", "method"]
    summary = by_task.groupby(by).mean().join(full.groupby(by)[["nomination_s", "total_s", "small_fits", "masks"]].sum())
    contrasts = []
    for panel in PHASES:
        t = by_task.loc[panel]
        for method in ("small", "slack"):
            delta = t.xs(method, level="method") - t.xs("point", level="method")
            idx = np.random.default_rng(33119).integers(len(delta), size=(30000, len(delta)))
            for field in fields:
                values = delta[field].to_numpy()
                lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
                contrasts.append(dict(panel=panel, method=method, reference="point", metric=field,
                                      difference=values.mean(), lo95=lo, hi95=hi))
    full.to_csv(DEST / "folds.csv", index=False)
    summary.to_csv(DEST / "summary.csv")
    pd.DataFrame(contrasts).to_csv(DEST / "contrasts.csv", index=False)
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("time", "replay"))
    if parser.parse_args().action == "time":
        timing()
    else:
        replay()
