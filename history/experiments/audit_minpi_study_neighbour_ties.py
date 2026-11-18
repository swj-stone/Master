"""Replay the measured reference choices and resolve distance ties across the full pool."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT
from experiments.minpi_study_raw_cost import panels
from minppi_mofs.piv3_compact_filter import _triplets
from minppi_mofs.word_minpi import exact_top_from_votes


def compare_triplets(ref, query, ref_group, group, weights, selections, comparison):
    distances = ((ref**2) @ weights)[None, :] - 2. * ((query * weights) @ ref.T)
    stable_hits, stable_misses = [], []
    archived_hits, archived_misses = [], []
    for i, label in enumerate(group):
        for same, archived, stable in ((True, archived_hits, stable_hits),
                                       (False, archived_misses, stable_misses)):
            pool = np.flatnonzero((ref_group == label) == same)
            depth = min(4, len(pool))
            values = distances[i, pool]
            head = pool[np.argpartition(values, depth-1)[:depth]]
            head = head[np.lexsort((head, distances[i, head]))]
            order = pool[np.lexsort((pool, values))]
            stable.extend([order[0], order[depth-1]])
            archived.extend([head[0], head[-1]])
            selections.append(dict(query=i, role="hit" if same else "miss", pool=len(pool),
                                   nearest_tied=int((values == values.min()).sum()) > 1,
                                   tie_across_depth=depth < len(pool) and distances[i, order[depth-1]] == distances[i, order[depth]],
                                   archived_first=head[0], archived_last=head[-1],
                                   sorted_first=order[0], sorted_last=order[depth-1]))
    q = np.repeat(np.arange(len(query)), 2)
    stable_margins = np.abs(query[q]-ref[stable_misses]) - np.abs(query[q]-ref[stable_hits])
    archived_margins = np.abs(query[q]-ref[archived_misses]) - np.abs(query[q]-ref[archived_hits])
    actual = _triplets(ref, query, ref_group, group, weights)
    np.testing.assert_array_equal(archived_margins, actual)
    comparison.append(np.sign(stable_margins).astype(np.int8))
    return actual


def audit(output):
    targets = pd.read_csv(OUTPUT/"training-verifier/five-arm/targets.csv")
    baseline = targets.loc[targets.budget_pct == 10].set_index(["source", "seed"])
    by_panel, by_query = [], []
    for source, seed, build, *_ in panels():
        if source == "mhealth" or (source, seed) not in baseline.index:
            continue
        selections, comparison = [], []
        def replay(ref, query, ref_group, group, weights):
            return compare_triplets(ref, query, ref_group, group, weights, selections, comparison)
        with patch("minppi_mofs.exact_minpi._triplets", replay):
            votes = build()
        alternative, = comparison
        target = baseline.loc[(source, seed), :]
        digest = sha256(votes.tobytes()).hexdigest()
        assert digest == target.votes_sha256
        chosen, _ = exact_top_from_votes(votes, int(target.h))
        np.testing.assert_array_equal(chosen, np.fromstring(str(target.support), dtype=int, sep=" "))
        stable, _ = exact_top_from_votes(alternative, int(target.h))
        frame = pd.DataFrame(selections).assign(source=source, seed=seed)
        changed = alternative != votes
        by_panel.append(dict(source=source, seed=seed, pools=len(frame),
                             nearest_ties=frame.nearest_tied.sum(), boundary_ties=frame.tie_across_depth.sum(),
                             different_references=((frame.archived_first != frame.sorted_first) |
                                                   (frame.archived_last != frame.sorted_last)).sum(),
                             changed_entries=changed.sum(), changed_rows=changed.any(axis=1).sum(),
                             support_swaps=len(set(stable)-set(chosen)), archived_sha256=digest,
                             sorted_sha256=sha256(alternative.tobytes()).hexdigest()))
        by_query.append(frame)
    assert len(by_panel) == len(baseline)-1 == 41
    q = np.array([0., 0.])
    hits = np.array([[1., 0.], [0., 1.]])
    miss = np.array([.6, .6])
    assert len(set(((hits-q)**2).sum(axis=1))) == 1
    toy = np.sign(np.abs(miss-q)-np.abs(hits-q)).astype(int).tolist()
    assert toy == [[-1, 1], [1, -1]]
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(by_panel).to_csv(output/"panels.csv", index=False)
    pd.concat(by_query, ignore_index=True).to_csv(output/"references.csv", index=False)
    (output/"example.json").write_text(json.dumps(dict(query=q.tolist(), hits=hits.tolist(),
        miss=miss.tolist(), possible_votes=toy, numpy=np.__version__), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(1):
        audit(parser.parse_args().output)
