"""Run each locked policy independently; this module has no outer outcome input."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as archived
from experiments.context_review.timing_process import FullBallot, bind_logical_cpu
from experiments.context_review.transport_data import EVIDENCE, ROOT, RUN, SPEC, read
from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import digest
from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.piv3_compact_tree import select as compact_gain
from minppi_mofs.piv3_compact_witness import _sample, _within

ARMS = ("gain", "vote", "slack", "point")


def panel_margin(after, before):
    margins = []
    for metric, reference in before.items():
        tolerance = (.05 if metric == "ap" else .03) * abs(reference)
        margin = (after[metric] - reference + tolerance + 1e-12) / max(tolerance, 1e-12)
        margins.append(margin)
    return min(margins)


def build_ballot(model, train, proposal, labels, seed, anchor, rare, point=False):
    task = "binary" if model.n_classes_ == 2 else "multiclass"
    ids, weights = _sample(labels, task, np.random.SeedSequence([seed, 659]))
    rows = np.random.default_rng(seed).choice(len(train), min(len(train), 512), replace=False)
    scorer = make_scorer(labels[ids], weights[:1] if point else weights, task, "ovr_auc", rare)
    return FullBallot(model, proposal[ids], np.median(train[rows], axis=0), frozenset(anchor), scorer)


def rating(ballot, cols, family, method, best=None):
    values = []
    for extension in family:
        after = ballot.score(frozenset(cols) | extension)
        before = ballot.score(ballot.anchor | extension)
        if method == "point":
            return panel_margin(after[0], before[0])
        if method == "vote":
            score = sum(_within(a, b) for a, b in zip(after, before, strict=True))
            if best is not None and (score <= best or score == 0):
                return score
        else:
            margins = sorted(panel_margin(a, b) for a, b in zip(after, before, strict=True))
            score = margins[1]  # second-smallest = third-largest of four
        values.append(score)
    return min(values)


def nominate(method, model, train, proposal, labels, seed, index, fold, order, k, rare):
    w = (k+1)//2
    anchor = tuple(sorted(order[:k]))
    if method == "gain" or w == k:
        return tuple(sorted(order[:w])), 0, []
    choices = archived.candidate_pool(order[:k], w, index, fold)[1]
    ballot = build_ballot(model, train, proposal, labels, seed, anchor, rare, point=method=="point")
    family = separating_contexts(order[k:]) if method != "point" else [frozenset()]
    incumbent, best = choices[0], -np.inf
    scored = []
    for cols in choices:
        value = rating(ballot, cols, family, method, best if method == "vote" else None)
        scored.append(dict(columns=cols, value=value))
        if value > best:
            best, incumbent = value, cols
        if method == "vote" and best == 4:
            break
    return incumbent, len(ballot.scores), scored


def policy(root, inputs, method, fold, index):
    x, y, outer, train, prop, guard = (inputs[key] for key in
                                       ("x", "y", "x_outer", "train", "proposal", "guard"))
    seed = 20421+fold
    start = perf_counter()
    transform = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    xt, xp = transform.transform(x[train]), transform.transform(x[prop])
    model = fit(xt, y[train], seed)
    task = "binary" if model.n_classes_ == 2 else "multiclass"
    rare = bool(task == "binary" and min(np.bincount(y[prop])) <= .05*len(prop))
    columns = [f"c{i}" for i in range(x.shape[1])]
    _, info = compact_gain(model, pd.DataFrame(xt, columns=columns), y[train],
                           pd.DataFrame(xp, columns=columns), y[prop], task=task, utility="ovr_auc",
                           seed=seed, aggregate="gain")
    order = [int(c[1:]) for c in info["order"]]
    k = int(info["k"])
    anchor = tuple(sorted(order[:k]))
    start_nomination = perf_counter()
    nominee, masks, precise = nominate(method, model, xt, xp, y[prop], seed, index, fold, order, k, rare)
    start_guard = perf_counter()
    passed, guard_scores, selected = False, {}, anchor
    if len(nominee) < k:
        inner = np.sort(np.r_[train, prop])
        probabilities, _, _ = archived.predict_sets(x, y, inner, x[guard], [anchor, nominee], seed)
        for arm, support in (("anchor", anchor), ("nominee", nominee)):
            guard_scores[arm] = archived.metrics(y[guard], probabilities[support], rare)
        passed = _within(guard_scores["nominee"], guard_scores["anchor"])
        if passed:
            selected = nominee
    end_guard = perf_counter()
    definition = dict(anchor=anchor, order=order, nominee=nominee, selected=selected,
                      passes=passed, guard=guard_scores, rare=rare)
    store(root / f"selection-{method}.json", definition)
    start_final = perf_counter()
    final, _, _ = archived.predict_sets(x, y, np.arange(len(x)), outer, [selected], seed)
    end_final = perf_counter()
    records = dict(wall_s=end_guard-start+end_final-start_final, nomination_s=start_guard-start_nomination,
                   guard_s=end_guard-start_guard, final_s=end_final-start_final,
                   reference_s=start_nomination-start,
                   initial_fits=1, guard_anchor_fits=int(nominee != anchor),
                   guard_candidate_fits=int(nominee != anchor), final_fits=1,
                   anchor_mask_checks=len(info["masked_proposals"]), masks=masks)
    return definition, final[selected], records, (precise, model, xt, xp)


def audited_pool(state, labels, fold, index, ballot_data):
    order, anchor, rare = state["order"], tuple(state["anchor"]), state["rare"]
    k = len(anchor)
    w = (k+1)//2
    if k == w:
        return []
    _, model, xt, xp = ballot_data
    candidates = archived.candidate_pool(order[:k], w, index, fold)[1]
    family = separating_contexts(order[k:])
    seed = 20421+fold
    ballot = build_ballot(model, xt, xp, labels, seed, anchor, rare)
    point_ballot = build_ballot(model, xt, xp, labels, seed, anchor, rare, point=True)
    rows = []
    for i, cols in enumerate(candidates):
        vote = rating(ballot, cols, family, "vote")
        slack = rating(ballot, cols, family, "slack")
        point = rating(point_ballot, cols, [frozenset()], "point")
        assert (slack >= 0) == (vote >= 3)
        rows.append(dict(columns=cols, gain=int(i==0), vote=vote, slack=slack, point=point))
    return rows


def predict_fold(name, fold, index, position, cpu, protocol):
    root = RUN / name / f"fold-{fold}"
    layout = read(root / "layout.json")
    assert digest(root / "inputs.npz") == layout["inputs_sha256"]
    selection = root / "selection.json"
    if selection.exists():
        prior = read(selection)
        assert prior["protocol"] == protocol
        assert prior["layout"] == digest(root / "layout.json")
        assert prior["predictions"] == digest(root / "predictions.npz")
        return
    with np.load(root / "inputs.npz") as data:
        inputs = {key: data[key] for key in data.files}
    schedule = ARMS[position:] + ARMS[:position]
    choices, probabilities, records = {}, {}, {}
    for arm in schedule:
        state, pred, times, internals = policy(root, inputs, arm, fold, index)
        choices[arm], probabilities[arm], records[arm] = state, pred, times
    assert all(v["order"] == state["order"] and v["anchor"] == state["anchor"] for v in choices.values())
    pool = audited_pool(state, inputs["y"][inputs["proposal"]], fold, index, internals)
    for arm in ARMS:
        if pool:
            winner = max(pool, key=lambda row: row[arm])["columns"]
            assert choices[arm]["nominee"] == winner, (name, fold, arm)
        for other in ARMS:
            if choices[arm]["selected"] == choices[other]["selected"]:
                np.testing.assert_array_equal(probabilities[arm], probabilities[other])
    anchor = state["anchor"]
    ref, _, _ = archived.predict_sets(inputs["x"], inputs["y"], np.arange(len(inputs["x"])),
                                      inputs["x_outer"], [anchor], 20421+fold)
    probabilities["anchor"] = ref[anchor]
    for arm in ARMS:
        if choices[arm]["selected"] == anchor:
            np.testing.assert_array_equal(probabilities[arm], probabilities["anchor"])
    np.savez_compressed(root / "predictions.npz", test_row=inputs["test_row"], **probabilities)
    store(selection, dict(protocol=protocol, layout=digest(root / "layout.json"),
                          predictions=digest(root / "predictions.npz"), schedule=schedule,
                          cpu=cpu, choices=choices, costs=records, pool=pool))


def predict():
    sources = [SPEC, EVIDENCE / "sources.json", Path(__file__),
               ROOT / "experiments/context_review/transport_split.py",
               ROOT / "experiments/context_review/transport_data.py",
               ROOT / "experiments/context_review/budgeted_guard.py",
               ROOT / "experiments/context_review/refit_target.py",
               ROOT / "experiments/context_review/timing_process.py",
               ROOT / "experiments/context_study/benchmark.py",
               ROOT / "experiments/test/minpi_convergence_metrics.py",
               ROOT / "src/minppi_mofs/context_design.py", ROOT / "src/minppi_mofs/piv3_compact_tree.py",
               ROOT / "src/minppi_mofs/piv3_compact_witness.py", ROOT / "src/minppi_mofs/utility.py"]
    path = RUN / "protocol.json"
    lock = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    if path.exists() and read(path) != lock:
        raise ValueError("Frozen input/source changed")
    store(path, lock)
    protocol = digest(path)
    cpu = bind_logical_cpu()
    receipts = {}
    for i, (name, number, _, _, _) in enumerate(read(SPEC)["sources"]):
        for fold in range(3):
            predict_fold(name, fold, 1000+number, (3*i+fold)%4, cpu, protocol)
            receipts[f"{name}/fold-{fold}"] = digest(RUN / name / f"fold-{fold}/selection.json")
        print(f"{name}: frozen all three sets of predictions, {len(receipts)}/30", flush=True)
    assert len(receipts) == 30
    store(EVIDENCE / "prediction-lock.json", dict(protocol=protocol, sources=lock, selections=receipts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with threadpool_limits(limits=1):
        predict()
