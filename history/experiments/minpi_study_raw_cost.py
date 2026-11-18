"""Time selection from the training arrays and audit natural exact budget boundaries."""
import argparse
import json
import platform
from fractions import Fraction
from functools import partial
from math import ceil, lcm
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.minpi_study import ROOT, OUTPUT, P, file_hash, load_dataset, train
from experiments.minpi_study_device import ballot, banks
from experiments.minpi_study_frontier import row_counts
from experiments.minpi_study_new_cohorts import SEEDS, load_cohort
from minppi_mofs.certified_minpi import _scores, _validate, _weights, top_from_votes
from minppi_mofs.exact_minpi import witness_margins
from minppi_mofs.integer_minpi import _layers, _rational_scores, rational_top_from_votes


def sizes(p):
    return sorted({min(k, p-1) for k in (0, 4, 16)}, reverse=True)


def first(scores, h):
    return np.argsort(-scores, kind="stable")[:h]


def integer_anchor_pruning(votes, h):
    """Upper-bound pruning with exact tallies everywhere and no fixed-point bounds."""
    v = _validate(votes, h)
    layers = _layers(v, sizes(v.shape[1]), None)
    common = lcm(*(layer[0] for layer in layers))
    upper = _rational_scores(v, layers[:1]) * (common // layers[0][0])
    head = np.sort(first(upper, h))
    if len(layers) == 1 or h == v.shape[1]:
        return head, {"refined": 0}
    other_multiple = common // lcm(*(layer[0] for layer in layers[1:]))

    def finish_columns(indices):
        secondary = _rational_scores(np.ascontiguousarray(v[:, indices]), layers[1:])
        return np.minimum(upper[indices], secondary * other_multiple)

    exact_head = finish_columns(head)
    candidates = upper >= min(exact_head)
    candidates[head] = False
    more = np.flatnonzero(candidates)
    if not len(more):
        return head, {"refined": len(head)}
    ids = np.sort(np.r_[head, more])
    upper[more] = finish_columns(more)
    upper[head] = exact_head
    return np.sort(ids[first(upper[ids], h)]), {"refined": len(ids)}


def largest(votes, h):
    v = _validate(votes, h)
    k = min(16, v.shape[1]-1)
    score = _scores(v, _weights(*row_counts(v), v.shape[1], [k])[k]) if k else v.mean(0)
    return np.sort(first(score, h)), {}


METHODS = {"float_full": partial(top_from_votes, prune=False), "float_cert": top_from_votes,
           "integer_full": partial(rational_top_from_votes, prune=False),
           "integer_anchor": integer_anchor_pruning, "integer_cert": rational_top_from_votes,
           "float_largest": largest}


def make_ballots(X, y, task, seed):
    return np.sign(witness_margins(X, y, task, seed)).astype(np.int8)


def make_device_ballots(X, ids):
    generators = np.random.SeedSequence(20261123).spawn(15)
    ref, query = banks(ids, np.arange(len(ids)), np.random.default_rng(generators[0]))
    scaler = StandardScaler().fit(X[ref].reshape(-1, 40))
    scaled = scaler.transform(X.reshape(-1, 40)).reshape(X.shape)
    return ballot(scaled, ids, ref, query, 16, np.random.default_rng(generators[1]), True)[0]


def downstream(xt, yt, xs, task, seed, cols):
    train("ridge", xt[:, cols], yt, xs[:, cols], task, seed)


def downstream_device(xt, yt, xs, cols):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=2000))
    model.fit(xt[:, cols], yt)
    for device in (0, 1):
        model.predict(xs[:, device][:, cols])


def panels():
    for name in P["primary_sources"]:
        X, y, task, *_ = load_dataset(name)
        for seed in P["formal_seeds"]:
            base = OUTPUT/"formal"/name/f"s{seed}"
            with np.load(base/"predictions.npz") as z:
                tr, te = z["train_indices"], z["test_indices"]
            record = json.loads((base/"record.json").read_text(encoding="utf-8"))
            xt, yt, xs = X[tr], y[tr], X[te]
            yield (name, seed, partial(make_ballots, xt, yt, task, seed),
                   partial(downstream, xt, yt, xs, task, seed),
                   None, np.asarray(record["arms"]["exact"]["selected"]))
    for name in ("gas", "supercon"):
        X, y, _, tr, te, task = load_cohort(name)
        xt, yt, xs = X[tr], y[tr], X[te]
        for seed in SEEDS:
            with np.load(OUTPUT/"new-cohorts"/f"{name}-{seed}.npz") as z:
                yield (name, seed, partial(make_ballots, xt, yt, task, seed),
                       partial(downstream, xt, yt, xs, task, seed),
                       z["train_votes"], z["columns_min"])
    with np.load(ROOT/"data/revision-cohorts/mhealth-windows.npz") as z:
        X, ids = z["windows"], z["ids"]
    with np.load(OUTPUT/"device/records.npz") as z:
        dev, te = z["dev"], z["test"]
        xt = X[dev]
        support = pd.read_csv(OUTPUT/"device/supports.csv").set_index("method").loc["minimum", "indices"]
        yield ("mhealth", 20261123, partial(make_device_ballots, xt, ids[dev]),
               partial(downstream_device, X[dev, z["train_device"]], ids[dev, 1], X[te]),
               z["train_votes"], np.fromstring(support, dtype=int, sep=" "))


def boundary(v, h):
    layers = _layers(v, sizes(v.shape[1]), 1 << 40)
    numerator = _rational_scores(v, layers)
    denominator = lcm(*(layer[0] for layer in layers))
    order = first(numerator, len(numerator))
    threshold = numerator[order[h-1]]
    gap = Fraction(threshold - numerator[order[h]], len(v) * denominator)
    error = max(layer[3] for layer in layers)
    near = sum(abs(score-threshold)*(1 << 40) <= 2*error*denominator for score in numerator)
    return np.sort(order[:h]), dict(gap_exact=str(gap), gap=float(gap), tied=int(sum(numerator == threshold)),
        within_twice_eta=int(near), eta=error/(len(v)*(1 << 40)),
        denominator_bits=max(layer[0].bit_length() for layer in layers)), numerator


def run(output):
    if output.exists():
        raise FileExistsError(output)
    trials, cases, fits, selections = [], [], [], []
    rng = np.random.default_rng(411217)
    for name, seed, build, fit, saved_votes, archived_minimum in panels():
        votes = build()
        if saved_votes is not None:
            np.testing.assert_array_equal(votes, saved_votes)
        w, p = votes.shape
        h = ceil(.1*p)
        exact, evidence, numerator = boundary(votes, h)
        computed = {method: fn(votes, h) for method, fn in METHODS.items()}
        for method in ("integer_full", "integer_anchor", "integer_cert"):
            np.testing.assert_array_equal(computed[method][0], exact)
        np.testing.assert_array_equal(computed["float_full"][0], computed["float_cert"][0])
        float_support = computed["float_cert"][0]
        integer_score_loss = sum(numerator[exact]) - sum(numerator[float_support])
        cases.append(dict(source=name, seed=seed, w=w, p=p, h=h, **evidence,
            float_changes=len(set(exact)-set(float_support)), float_loses_score=integer_score_loss > 0,
            archived_changes=len(set(archived_minimum)-set(exact)),
            anchor_refined=computed["integer_anchor"][1]["refined"],
            **computed["integer_cert"][1]))
        for method, (cols, _) in computed.items():
            selections.append(dict(source=name, seed=seed, method=method, support=" ".join(map(str, cols))))
        for r in range(-1, 5):
            for position, method in enumerate(rng.permutation(list(METHODS))):
                start = perf_counter_ns()
                v = build()
                made = perf_counter_ns()
                selected, _ = METHODS[method](v, h)
                stop = perf_counter_ns()
                np.testing.assert_array_equal(votes, v)
                np.testing.assert_array_equal(selected, computed[method][0])
                if r >= 0:
                    trials.append(dict(source=name, seed=seed, method=method, repeat=r, position=position,
                        construct_ms=(made-start)/1e6, score_ms=(stop-made)/1e6, select_ms=(stop-start)/1e6))
            start = perf_counter_ns()
            fit(exact)
            stop = perf_counter_ns()
            if r >= 0:
                fits.append(dict(source=name, seed=seed, repeat=r, downstream_ms=(stop-start)/1e6))
    output.mkdir(parents=True)
    for name, rows in (("cases", cases), ("trials", trials), ("fit", fits), ("supports", selections)):
        pd.DataFrame(rows).to_csv(output/f"{name}.csv", index=False)
    inputs = [Path(__file__), ROOT/"docs/minpi-study/raw-cost-protocol.md",
              ROOT/"src/minppi_mofs/integer_minpi.py", ROOT/"src/minppi_mofs/certified_minpi.py",
              ROOT/"src/minppi_mofs/exact_minpi.py", ROOT/"experiments/minpi_study_device.py",
              ROOT/"experiments/minpi_study.py", ROOT/"experiments/minpi_study_new_cohorts.py"]
    manifest = dict(sources={p.relative_to(ROOT).as_posix(): file_hash(p) for p in inputs},
                    numpy=np.__version__, python=platform.python_version(), threads=1, repeats=5)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(1):
        run(parser.parse_args().output)
