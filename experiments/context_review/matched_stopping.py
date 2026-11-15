"""Exact matched stopping for ordinal, continuous and lexicographic forest nominations."""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from threading import Timer
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import filter_neighborhood, forest_predict, neighborhood_replicates
from experiments.context_review import transport_predict as primary
from experiments.context_review.forest_replay import read_member
from experiments.context_review.receiver_transport import SOURCE
from minppi_mofs.piv3_compact_witness import _within


DEST = bg.ROOT / "thesis-paper/evidence/matched_stopping"
DESIGN = bg.ROOT / "configs/context_review/matched_stopping.json"
RUN = bg.ROOT / "results/context_review/matched-stopping"
ARMS = [(method, prune) for method in ("vote", "margin", "vote_margin") for prune in (False, True)]
ARMS.append(("point", False))


def choose(options, family, evaluate, method, prune):
    """Maximize a coordinatewise context minimum, retaining the first exact tie.

    evaluate(candidate, context) returns either one score or a (vote, margin)
    tuple. All unopened contexts can only decrease either running minimum.
    """
    if not options:
        return 0
    winner, best = 0, None
    for candidate in options:
        upper = None
        for restoration in family:
            score = evaluate(candidate, restoration)
            upper = score if upper is None else tuple(map(min, upper, score))
            if prune and best is not None and upper <= best:
                break
            if prune and method == "vote" and upper == (0,):
                break
        if best is None or upper > best:
            winner, best = candidate, upper
            if prune and method == "vote" and best == (4,):
                break
    return winner


def context_score(after, before, method):
    if method == "point":
        return (primary.panel_margin(after[0], before[0]),)
    if method != "margin":
        votes = sum(_within(a, b) for a, b in zip(after, before, strict=True))
        if method == "vote":
            return (votes,)
    margins = sorted(primary.panel_margin(a, b) for a, b in zip(after, before, strict=True))
    if method == "margin":
        return (margins[1],)
    return votes, margins[1]


@dataclass
class Slate:
    design: str
    draw: int
    slates: list
    family: tuple
    indices: list
    metrics: list
    winners: dict

    def path(self, method):
        return self.family[:1] if method == "point" else self.family

    def evaluate(self, method, score):
        def evaluate(candidate, restoration):
            after = score(self.slates[candidate] | restoration)
            before = score(self.slates[0] | restoration)
            return context_score(after, before, method)
        return evaluate


def folds():
    g = bg.read(forest_predict.DEST / "prediction-lock.json")
    a = bg.read(filter_neighborhood.DEST / "prediction-lock.json")
    r = bg.read(neighborhood_replicates.DEST / "prediction-lock.json")
    assert g["folds"].keys() == a["folds"].keys() == r["folds"].keys()
    for key, rec in g["folds"].items():
        src = bg.ROOT / "results/context_review" / SOURCE[rec["panel"]] / key.split("/", 1)[1]
        assert bg.digest(src/"selection.json") == rec["selection_sha"]
        selection = bg.read(src/"selection.json")
        local, repeats = a["folds"][key], r["folds"][key]
        slates = []
        for design, source, meta, choices, winners, order in (
                ("gain", forest_predict.DEST, rec, [list(range(len(rec["supports"])))],
                 [rec["winners"]], selection["choices"]["gain"]["order"]),
                ("anova", neighborhood_replicates.DEST, repeats, repeats["slates"], repeats["winners"], local["order"])):
            sets = list(map(frozenset, meta["supports"]))
            family = primary.separating_contexts(order[len(sets[0]):])
            for draw, (ids, win) in enumerate(zip(choices, winners, strict=True), 1):
                assert ids[0] == 0 and all(not set(order[len(sets[0]):]) & sets[i] for i in ids)
                slates.append((source, meta, Slate(design, draw, sets, family, ids[1:], meta["metrics"], win)))
        yield key, src, rec, slates


def identifiers(key, slate, method, pruned):
    panel, dataset, fold = key.split("/")
    return dict(panel=panel, dataset=dataset, fold=int(fold.split("-")[1]), design=slate.design,
                draw=slate.draw, method=method, pruned=pruned)


def check_locked(slate, method, nominee):
    key = {"vote": "vote", "margin": "slack", "point": "point"}.get(method)
    if key is not None:
        assert nominee == slate.winners[key], (slate.design, slate.draw, key, nominee, slate.winners[key])


def frozen_rows(key, slate, arrays):
    names = slate.metrics
    tensor = arrays["tensor"]
    # Duplicate restoration sets can give the same visible mask. Maintain the
    # actual cache identity across BOTH anchor and candidate. Do not bill a tuple
    # (candidate, context) separately when it represents the same prediction.
    tables = {}
    if slate.indices:
        for i in [0, *slate.indices]:
            for c, ext in enumerate(slate.family):
                mask = slate.slates[i] | ext
                val = tensor[i, c]
                if mask in tables:
                    np.testing.assert_array_equal(val, tables[mask])
                else:
                    tables[mask] = val
    rows, exhausted = [], {}
    for method, pruned in ARMS:
        seen = set()

        @cache
        def fetch(mask):
            seen.add(mask)
            return [dict(zip(names, row, strict=True)) for row in tables[mask]]

        evaluate = slate.evaluate(method, fetch)
        if method == "point" and slate.design == "anova":
            # The single-panel scorer can round differently to a batched one.
            point_values = arrays["points"]

            def evaluate(i, e):
                fetch(slate.slates[0] | e)
                fetch(slate.slates[i] | e)
                return (point_values[i-1],)
        nominee = choose(slate.indices, slate.path(method), evaluate, method, pruned)
        check_locked(slate, method, nominee)
        if pruned:
            assert nominee == exhausted[method]
        else:
            exhausted[method] = nominee
        rows.append(dict(**identifiers(key, slate, method, pruned), winner=nominee, masks=len(seen)))
    return rows


def replay():
    DEST.mkdir(exist_ok=True, parents=True)
    rows = []
    with ExitStack() as stack:
        archives = {d:stack.enter_context(ZipFile(d/"predictions.zip"))
                    for d in (forest_predict.DEST, neighborhood_replicates.DEST)}
        for key, _, _, slates in folds():
            for path, meta, slate in slates:
                with read_member(archives[path], key, meta["member_sha"]) as values:
                    rows.extend(frozen_rows(key, slate, values))
    folds_table = pd.DataFrame(rows)
    assert len(folds_table) == 184*7
    folds_table.to_csv(DEST/"folds.csv", index=False)
    group = ["panel", "design", "draw", "method", "pruned"]
    summary = folds_table.groupby(group, as_index=False).agg(masks=("masks", "sum"), folds=("fold", "size"))
    summary.to_csv(DEST/"summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    if (DEST/"timings.csv").exists():
        summarize_times(folds_table)


def time_fold(index, key, src, rec, slates):
    assert bg.digest(src/"inputs.npz") == rec["inputs_sha"]
    with np.load(src/"inputs.npz") as data:
        x, y, train, prop = [data[k] for k in ("x", "y", "train", "proposal")]
    xt, xp = forest_predict.transform(x, train, x[prop])
    seed = 20421 + rec["fold"]
    voter = forest_predict.new_forest(seed).fit(xt, y[train])
    assert np.array_equal(voter.classes_, np.arange(len(voter.classes_)))
    records = []
    for slate_index, (_, _, slate) in enumerate(slates):
        for repeat in range(3):
            order = np.random.default_rng(np.random.SeedSequence([37189,index,slate_index,repeat])).permutation(len(ARMS))
            for position, arm_index in enumerate(order):
                method, prune = ARMS[arm_index]
                start = perf_counter()
                ballot = primary.build_ballot(voter, xt, xp, y[prop], seed, slate.slates[0],
                                              "ap" in rec["metrics"], point=method=="point")
                nominee = choose(slate.indices, slate.path(method), slate.evaluate(method,ballot.score), method, prune)
                elapsed = perf_counter()-start
                check_locked(slate, method, nominee)
                records.append(dict(**identifiers(key,slate,method,prune), repeat=repeat+1, position=position+1,
                                    winner=nominee, masks=len(ballot.scores), nomination_seconds=elapsed))
    return records


def summarize_times(fold_table):
    times = pd.read_csv(DEST/"timings.csv")
    key = ["panel","dataset","fold","design","draw","method","pruned"]
    assert len(times)==3*len(fold_table) and not times.duplicated(key+["repeat"]).any()
    agreement = times.merge(fold_table,on=key,validate="many_to_one",suffixes=("_live","_stored"))
    assert len(agreement)==len(times)
    for col in ("winner","masks"):
        np.testing.assert_array_equal(agreement[col+"_live"], agreement[col+"_stored"])
    medians = times.groupby(key,as_index=False).nomination_seconds.median()
    total = medians.groupby(["panel","design","draw","method","pruned"],as_index=False).nomination_seconds.sum()
    total.to_csv(DEST/"timing-summary.csv", index=False)
    print(total.to_string(index=False), flush=True)


def timed():
    DEST.mkdir(exist_ok=True,parents=True)
    digest = bg.digest(Path(__file__))
    rows = []
    began = perf_counter()
    with threadpool_limits(limits=1):
        for index,(key,src,rec,slates) in enumerate(folds()):
            path = RUN/key/"times.json"
            path.parent.mkdir(exist_ok=True, parents=True)
            if path.exists():
                result = bg.read(path)
                assert result["code"]==digest and result["design"]==bg.digest(DESIGN)
            else:
                timer = Timer(3600, forest_predict.deadline, args=(key,))
                timer.daemon = True
                timer.start()
                try:
                    new_rows = time_fold(index,key,src,rec,slates)
                    result = dict(code=digest,design=bg.digest(DESIGN),rows=new_rows)
                    stage = path.with_suffix(".tmp")
                    bg.store(stage,result)
                    stage.replace(path)
                finally:
                    timer.cancel()
            rows.extend(result["rows"])
            elapsed = perf_counter()-began
            print(f"[{index+1}/46] {key}, elapsed {elapsed:.1f}s ETA {elapsed*(45-index)/(index+1):.1f}s",flush=True)
    pd.DataFrame(rows).to_csv(DEST/"timings.csv", index=False)
    summarize_times(pd.read_csv(DEST/"folds.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("replay","time"), default="replay", nargs="?")
    if parser.parse_args().action == "time":
        timed()
    else:
        replay()
