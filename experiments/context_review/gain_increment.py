"""Matched Gain shortlists with optional minimum voting and uniform or point tie-breaking."""
import argparse
from math import ceil, fsum
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import transport_predict as voter
from experiments.context_review.decision_opportunities import EVIDENCE, PHASES, RUNS


DESIGN = bg.ROOT / "configs/context_review/gain_increment.json"
DEST = EVIDENCE / "gain_increment"
POLICIES = ("uniform", "vote_uniform", "point", "vote_point")
FRACTIONS = {"quarter": .25, "half": .5, "three_quarters": .75, "all": 1.}


def maximum(items, score):
    best = max(score[i] for i in items)
    return [i for i in items if score[i] == best]


def distributions(sets, votes, points):
    """Lists are ordered by ascending rank sum and then lexicographic support."""
    groups = {}
    for name, indices in sets.items():
        if indices == [0]:
            groups[name] = {p: [0] for p in POLICIES}
            continue
        tied = maximum(indices, votes)
        groups[name] = {
            "uniform": indices, "vote_uniform": tied,
            "point": maximum(indices, points)[:1], "vote_point": maximum(tied, points)[:1],
        }
    return groups


def define_choices(selection, supports, gain):
    pool = selection["pool"]
    order = selection["choices"]["gain"]["order"]
    ids = [supports.index(row["columns"]) for row in pool]
    ranks = {c: i for i, c in enumerate(order)}
    negative_rank_sum = {i: -sum(ranks[c] for c in supports[i]) for i in ids}
    gain_mass = {i: fsum(gain[c] for c in supports[i]) for i in ids}
    assert ids == sorted(ids, key=lambda i: (-negative_rank_sum[i], supports[i]))
    assert sorted(ids) == list(range(1, len(supports)))
    votes = {i: row["vote"] for i, row in zip(ids, pool, strict=True)}
    points = {i: row["point"] for i, row in zip(ids, pool, strict=True)}
    sets = {k: ids[:ceil(frac * len(ids))] if ids else [0] for k, frac in FRACTIONS.items()}
    sets["strict_rank"] = maximum(ids, negative_rank_sum) if ids else [0]
    sets["strict_mass"] = maximum(ids, gain_mass) if ids else [0]
    gain_nominee = supports.index(selection["choices"]["gain"]["nominee"])
    assert sets["strict_rank"] == [gain_nominee] and gain_nominee in sets["strict_mass"]
    return sets, distributions(sets, votes, points)


def time_primary(policy, ids, supports, make_ballot, family):
    start = perf_counter()
    if len(ids) == 1 or policy == "uniform":
        return ids, dict(masks=0, seconds=perf_counter()-start)
    ballot = make_ballot(policy == "point")
    if policy in ("point", "vote_point"):
        point = {i: voter.rating(ballot, supports[i], [frozenset()], "point") for i in ids}
        ids = sorted(ids, key=lambda i: -point[i])
        if policy == "point":
            return ids[:1], dict(masks=len(ballot.scores), seconds=perf_counter()-start)
    ties, incumbent = [], -1
    for i in ids:
        # A uniform lottery must recover every maximizer. A deterministic
        # point tie-break has already ordered candidates from best to worst.
        cutoff = incumbent-1 if policy == "vote_uniform" else incumbent
        value = voter.rating(ballot, supports[i], family, "vote", best=cutoff)
        if value > incumbent:
            incumbent, ties = value, [i]
        elif policy == "vote_uniform" and value == incumbent:
            ties.append(i)
        if policy == "vote_point" and incumbent == 4:
            break
    return ties, dict(masks=len(ballot.scores), seconds=perf_counter()-start)


def selection_run(path, selection, rec, position):
    layout = bg.read(path / "layout.json")
    assert selection["layout"] == bg.digest(path / "layout.json")
    assert layout["inputs_sha256"] == bg.digest(path / "inputs.npz")
    with np.load(path / "inputs.npz") as data:
        train, prop = data["train"], data["proposal"]
        x, y = data["x"], data["y"]
    tr = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    xt, xp = tr.transform(x[train]), tr.transform(x[prop])
    seed = 20421 + rec["fold"]
    model = voter.fit(xt, y[train], seed)
    gains = {int(key[1:]): v for key, v in model.get_booster().get_score(importance_type="gain").items()}
    order = selection["choices"]["gain"]["order"]
    assert order == (sorted(gains, key=lambda i: (-gains[i], i)) or [0])
    anchor = tuple(selection["choices"]["gain"]["anchor"])
    assert set(anchor) == set(order[:len(anchor)])
    assert bool(selection["pool"]) == (len(anchor) > 1)
    assert list(anchor) == rec["supports"][0]
    gain = [gains.get(i, 0.) for i in range(x.shape[1])]
    short, groups = define_choices(selection, rec["supports"], gain)

    def make_ballot(point_only):
        return voter.build_ballot(model, xt, xp, y[prop], seed, anchor, rec["rare"], point_only)

    family = voter.separating_contexts(order[len(anchor):])
    start = position % len(POLICIES)
    times = {}
    for p in POLICIES[start:]+POLICIES[:start]:
        nominees, times[p] = time_primary(p, short["half"], rec["supports"], make_ballot, family)
        assert nominees == groups["half"][p], (rec["dataset"], rec["fold"], p, nominees)
    return dict(gain=gain, shortlists=short, policies=groups, primary_cost=times,
                selection_sha=rec["selection_sha"], inputs_sha=layout["inputs_sha256"])


def measure(output):
    if output.exists():
        raise FileExistsError(f"Use a new output file to time a fresh pass: {output}")
    records = bg.read(EVIDENCE / "receiver/prediction-lock.json")["folds"]
    assert len(records) == 46
    lock = dict(design=bg.digest(DESIGN), folds={})
    with threadpool_limits(limits=1):
        for position, (name, rec) in enumerate(sorted(records.items())):
            panel, key = name.split("/", 1)
            path = RUNS / PHASES[panel][0] / key
            assert bg.digest(path / "selection.json") == rec["selection_sha"]
            lock["folds"][name] = selection_run(path, bg.read(path / "selection.json"), rec, position)
            print(f"[{position+1}/46] {name}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    bg.store(output, lock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEST / "nomination-lock.json")
    measure(parser.parse_args().output)
