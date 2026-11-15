"""Check the exact distinct-mask cost and replay all context orders at fixed decisions."""
import csv
from itertools import permutations
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1] / "results/thesis_benchmark/context-vote-20261115"


def cost(prefixes):
    # Every visited context also needs its anchor reference, shared by all candidates.
    return sum(prefixes) + max(prefixes, default=0)


def replay(diag):
    anchor = frozenset(diag["anchor_order"])
    live = anchor
    votes = {}
    for proposal in diag["trace"]:
        target = live.difference(proposal["block"])
        if not target or not target < live or not target < anchor:
            raise ValueError("The cache identity requires a proper, nonempty deletion")
        ballot = tuple(sum(panel) >= 3 for panel in proposal["votes"])
        if votes.setdefault(target,ballot) != ballot:
            raise ValueError("A repeated candidate changed ballots")
        if all(ballot) != proposal["accepted"]:
            raise ValueError("A recorded decision differs from the context predicate")
        if proposal["accepted"]:
            live = target
    measured = diag["witness_predictions"]
    if cost([len(p) for p in votes.values()]) != measured:
        raise ValueError("Distinct masks differ from the cache identity")
    return votes, measured


def main():
    with (RUN/"per-seed-effective.csv").open(encoding="utf-8-sig",newline="") as stream:
        records = [r for r in csv.DictReader(stream) if r["arm"] in ("min","eager")]
    summary = []
    for r in records:
        diag=json.loads((RUN/r["run_folder"]/"estimator.json").read_bytes())
        votes,measured=replay(diag)
        h=len(diag["context_extensions"])
        eager=h*(len(votes)+1) if votes else 0
        out=dict(dataset=r["dataset"],seed=r["seed"],arm=r["arm"],contexts=h,
                 distinct_candidates=len(votes),measured_masks=measured,eager_masks=eager)
        if r["arm"]=="eager":
            if eager!=measured or any(len(p)!=h for p in votes.values()):
                raise ValueError("The eager trace is incomplete")
            costs=[]
            for order in permutations(range(h)):
                lengths=[next((i+1 for i,j in enumerate(order) if not p[j]),h) for p in votes.values()]
                costs.append(cost(lengths))
            out.update(original_order=costs[0],best_order=min(costs),worst_order=max(costs))
        summary.append(out)
    if len(summary)!=36:
        raise ValueError("Incomplete mask audit")
    for e in (t for t in summary if t["arm"]=="eager"):
        lazy=next(x for x in summary if x["arm"]=="min" and x["seed"]==e["seed"] and x["dataset"]==e["dataset"])
        if lazy["measured_masks"]!=e["original_order"]:
            raise ValueError("Eager replay disagrees with observed short-circuit calls")
    with (HERE/"mask-cost.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,list(dict.fromkeys(k for row in summary for k in row)))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Exact mask cost verified for all {len(summary)} trajectories")


if __name__=="__main__":
    main()
