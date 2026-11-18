"""Attribute low-dimensional score changes to width, initialization and block additions."""
import argparse
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.block_selection.fresh import fresh_cases
from research.feature_game_eval import panels as bench
from research.lowdim.method import block_support, fast_support, pair_support
from research.pair_checks.reference import Refits, standardize
from research.pair_selection.benchmark import INDEX, KEY, cases, external_cases


HERE = Path(__file__).resolve().parent


def penalized_score(ref, m, total, prefix, suffix):
    loss = total - ref.utility(prefix)
    after = total - ref.utility(list(prefix) + list(suffix))
    return float(m*np.log(loss/after) - len(suffix)*np.log(m))


def supports(case, key):
    X = bench._impute_validation(case.X, case.X[:0])[0] if case.impute else case.X
    options = dict(task=case.task, seed=key[-1])
    vote, info = fast_support(X, case.y, **options)
    pair, _ = pair_support(X, case.y, **options)
    block, audit = block_support(X, case.y, audit=True, **options)
    width, cap = len(block), len(vote)
    assert cap == audit["width_cap"] >= width
    z, response = standardize(X, case.y, case.task, key[-1])
    ref = Refits(z, response)
    total = float(np.sum(response**2))
    m, p = z.shape
    anchors = ((j,) for j in range(p))
    anchor = ref.best((), anchors)
    seed = ref.best_pair() if cap >= 2 else anchor
    cap_path = ref.forward(seed, cap)
    np.testing.assert_array_equal(sorted(cap_path), pair)
    vote_path = np.argsort(-info["rank"], kind="stable")
    np.testing.assert_array_equal(sorted(vote_path[:cap]), vote)
    stop_path = list(seed)
    if not len(audit["block_sizes"]):
        stop_path = list(block)
    while len(stop_path) < cap:
        addition = ref.best(stop_path, ((j,) for j in range(p) if j not in stop_path))
        if penalized_score(ref, m, total, stop_path, addition) <= 0:
            break
        stop_path.extend(addition)
    arms = dict(block=block, pair_cap=cap_path, vote_cap=vote,
                pair_matched=cap_path[:width], anchor_matched=ref.forward(anchor, width),
                vote_matched=vote_path[:width], singleton_stop=stop_path)
    diagnostics = []
    prefix = []
    order = audit["selection_order"].tolist()
    for size in audit["block_sizes"].tolist():
        suffix = order[len(prefix):len(prefix) + size]
        if prefix and size == 2:
            greedy = ref.forward(prefix, len(prefix)+2)[-2:]
            top_single = penalized_score(ref, m, total, prefix, greedy[:1])
            diagnostics.append(dict(
                prefix=" ".join(map(str, prefix)), chosen=" ".join(map(str, suffix)),
                greedy=" ".join(map(str, greedy)), same_set=set(suffix) == set(greedy),
                includes_best_single=greedy[0] in suffix,
                best_single_score=top_single,
                chosen_block_score=penalized_score(ref, m, total, prefix, suffix),
                greedy_block_score=penalized_score(ref, m, total, prefix, greedy)))
        prefix.extend(suffix)
    return arms, ref, p, width, cap, diagnostics


def run(timed):
    archive = pd.read_csv(INDEX, keep_default_na=False).query("p <= 32").set_index(KEY)
    want = set(archive.index)
    output = []
    pairs = []
    prior = pd.concat([
        pd.read_csv(timed/f"{suite}-pass1/quality.csv", keep_default_na=False)
        for suite in ("archive", "external", "fresh")
    ]).set_index(KEY + ["method"])
    for case in chain((c for c in cases(want) if c.key in want), external_cases(), fresh_cases()):
        key = case.key
        meta = dict(zip(KEY, key))
        arms, ref, p, width, cap, joint = supports(case, key)
        for item in joint:
            pairs.append(meta | item)
        # Freeze all seven training-only supports before accessing held-out predictions.
        observed = {}
        for arm, path in arms.items():
            ids = tuple(sorted(path))
            if ids not in observed:
                observed[ids] = case.score(*case.predict(np.array(ids, dtype=np.int64)))
            columns = " ".join(map(str, ids))
            official = dict(block="block", pair_cap="pair", vote_cap="current").get(arm)
            if official:
                timed_row = prior.loc[key + (official,)]
                assert columns == str(timed_row.cols) and len(ids) == timed_row.k
                np.testing.assert_allclose(observed[ids], timed_row.score, atol=1e-12, rtol=1e-12)
            output.append(meta | dict(metric=prior.loc[key+("block",)].metric,
                                      p=p, cap=cap, block_width=width, arm=arm,
                                      k=len(ids), columns=columns,
                                      utility=ref.utility(ids), score=observed[ids]))
    result = pd.DataFrame(output).sort_values(KEY + ["arm"])
    assert len(result) == 74*7 and not result.duplicated(KEY + ["arm"]).any()
    assert (result.cohort == "fresh").sum() == 6*7
    result.to_csv(HERE/"cases.csv", index=False)
    pd.DataFrame(pairs).sort_values(KEY, kind="stable").to_csv(HERE/"residual-pairs.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timed_run", type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.timed_run)
