"""Falsify the proposed pair-margin model using the archived mask predictions."""
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.context_diagnostics.design import digest
from experiments.test.minpi_convergence_metrics import make_scorer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = ROOT / "results/thesis_benchmark/context-mechanism-r1/trees"


def margins(predictions, y, weights, rec):
    score = make_scorer(y, weights, rec["task"], rec["utility"], rec["rare"])
    scores = [score(p) for p in predictions]
    cols = list(scores[0][0])
    values = np.array([[[b[key] for key in cols] for b in s] for s in scores])
    index = np.asarray(rec["lookup"])
    before = values[index[0]]
    after = values[index[1:]]
    direction = np.array([-1 if key == "mae" else 1 for key in cols])
    tau = np.array([.05 if key=="ap" else .03 for key in cols])
    loss = (before-after)*direction
    thresholds = np.abs(before)*tau + 1e-12
    panel = np.max(loss-thresholds, axis=-1)
    g = np.sort(panel, axis=-1)[:, :, -2]
    assert np.array_equal(g > 0, (loss > thresholds).any(axis=-1).sum(axis=-1) >= 2)
    return g


def fits_margin(g, z):
    # WLOG widen the upper endpoint U=a+eta to zero while holding L=a-eta
    # fixed. The remaining scalar inequalities are maximized at delta=-max(g_-).
    negative, middle, positive = (g[z==j] for j in (-1, 0, 1))
    delta = -negative.max()
    return (middle.max() <= 0 and positive.min() > 0 and positive.max() <= delta
            and delta > max(0., -middle.min(), -.5*negative.min()))


def analyse(path):
    rec = json.loads(path.read_bytes())
    tail = sorted(rec["tail"])
    if len(tail) < 2:
        return []
    values = path.parent/"predictions.npz"
    assert digest(values) == rec["predictions_sha256"]
    pairs = np.asarray(list(permutations(range(len(tail)), 2)))
    passes = np.asarray([[i in mask for i in tail] for mask in rec["extensions"]], dtype=np.int8)
    z = passes[:,pairs[:,0]] - passes[:,pairs[:,1]]
    assert (z==1).any(axis=0).all() and (z==-1).any(axis=0).all()
    endpoints = [i for i,m in enumerate(rec["extensions"]) if not m or set(m)==set(tail)]
    assert len(endpoints) == 2
    with np.load(values, allow_pickle=False) as data:
        preds = data["predictions"]
        n = len(data["rank_y"])
        patterns = {side:margins(part, data[f"{side}_y"], data[f"{side}_w"], rec)
                    for side, part in (("rank",preds[:,:n]), ("guard",preds[:,n:]))}
    results = []
    for c in range(len(rec["candidates"])):
        row = dict(dataset=rec["dataset"], seed=rec["seed"], candidate=c,
                   tail=len(tail), masks=len(passes))
        previous = {}
        for side in patterns:
            g = patterns[side][c]
            signs = g > 0
            signature_matches = np.flatnonzero(((z==1) == signs[:,None]).all(axis=0))
            margin_matches = [i for i in signature_matches if fits_margin(g,z[:,i])]
            for label, subset in (("sign",signature_matches), ("margin",margin_matches)):
                candidates = set(subset)
                row[f"{side}_{label}_pairs"] = len(candidates)
                if side == "guard":
                    row[f"both_{label}_pairs"] = len(previous[label] & candidates)
                previous[label] = candidates
            row[f"{side}_vetoes"] = int(signs.sum())
            row[f"{side}_endpoint_veto"] = bool(signs[endpoints].any())
            row[f"{side}_novel_failing"] = int(signs[rec["novel"]].sum())
        results.append(row)
    return results


if __name__ == "__main__":
    frame = pd.DataFrame(r for path in sorted(INPUT.glob("seed-*/*/receipt.json")) for r in analyse(path))
    # Abalone has one split with two tail columns and all four possible masks;
    # it is informative for this test even though no unqueried mask remains.
    assert len(frame)==226 and frame.dataset.nunique()==6
    old = pd.read_csv(INPUT/"decisions.csv").query("rule == 'ends'")
    checked = frame.merge(old, left_on=["dataset","seed","candidate"],
                          right_on=["dataset","seed","proposal"], validate="one_to_one")
    assert len(checked)==len(frame)
    for side, accepted in (("rank","accept"),("guard","guard_design_accept")):
        assert np.array_equal(checked[f"{side}_endpoint_veto"], checked[accepted].eq(0))
        field = f"{side}_novel_failing"
        assert np.array_equal(checked[f"{field}_x"], checked[f"{field}_y"])
    frame.to_csv(HERE/"pair-model-proposals.csv", index=False)
    summaries=[]
    for ds, group in [*frame.groupby("dataset"), ("all",frame)]:
        for side in ("rank","guard"):
            g=group.copy()
            n=g[f"{side}_vetoes"]
            endpoints=g[f"{side}_endpoint_veto"]
            result=dict(dataset=ds,side=side,proposals=len(g),all_pass=int((n==0).sum()),
                        all_veto=int((n==g.masks).sum()),mixed=int(((n>0)&(n<g.masks)).sum()),
                        endpoint_veto=int(endpoints.sum()),endpoint_pass_interior_veto=int(((~endpoints)&(n>0)).sum()))
            for label in ("sign","margin"):
                result[label+"_compatible"] = int((g[f"{side}_{label}_pairs"]>0).sum())
                result["same_pair_both_"+label] = int((g[f"both_{label}_pairs"]>0).sum())
            summaries.append(result)
    table = pd.DataFrame(summaries)
    table.to_csv(HERE/"pair-model-summary.csv",index=False)
    print(table.to_string(index=False))
