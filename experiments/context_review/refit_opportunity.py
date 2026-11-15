"""Refit every candidate in the archived half-width slates to locate missed opportunities."""
import argparse
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg

DESIGN = bg.ROOT / "configs/context_review/refit_opportunity.json"
OUT = bg.ROOT / "results/context_review/refit-opportunity"
EVIDENCE = bg.ROOT / "thesis-paper/evidence/opportunity"
POLICIES = ("fraction_gain", "vote_gain")


def outcome(scores, reference, tolerance):
    return all(reference[key] - value <= (0.05 if key == "ap" else tolerance) *
               abs(reference[key]) + 1e-12 for key, value in scores.items())


def replay_saved(folder, measures, stored_supports, archive_name):
    with np.load(folder / archive_name) as previous:
        for index, support in enumerate(stored_supports):
            if support in measures:
                np.testing.assert_array_equal(measures[support], previous[f"s{index}"])


def fit_fold(dataset, fold):
    previous = bg.RUN / dataset / f"fold-{fold}"
    dest = OUT / dataset / f"fold-{fold}"
    dest.mkdir(parents=True, exist_ok=True)
    slate = bg.read(previous / "shortlists.json")
    selection = bg.read(previous / "selection.json")
    check = bg.read(previous / "completed.json")
    if selection["shortlists"] != bg.digest(previous / "shortlists.json") or check["selection"] != bg.digest(previous / "selection.json"):
        raise ValueError("The archived slate or prior selection changed")
    anchor = tuple(slate["anchor"])
    width = (len(anchor) + 1) // 2
    candidates = [tuple(s) for s in slate["pools"].get(str(width), [[]])[0]]
    supports = [anchor, *candidates]
    assert len(set(supports)) == len(supports)
    x, y, train, proposal, guard, dev, future, get_target, _, data_hash = bg.inputs(dataset, fold)
    assert data_hash == slate["source"]
    for key, ids in (("train", train), ("proposal", proposal), ("guard", guard), ("dev", dev)):
        np.testing.assert_array_equal(ids, slate[key])
    inner = np.sort(np.r_[train, proposal])
    g, _, _ = bg.predict_sets(x, y, inner, x[guard], supports, slate["seed"])
    o, _, _ = bg.predict_sets(x, y, dev, future, supports, slate["seed"])
    replay_saved(previous, g, [tuple(s) for s in selection["supports"]], "guard.npz")
    old_finals = sorted({tuple(s) for s in selection["selections"].values()})
    replay_saved(previous, o, old_finals, "predictions.npz")
    target = get_target()
    probabilities = {f"{phase}{i}": probs[s] for i, s in enumerate(supports) for phase, probs in (("g", g), ("o", o))}
    np.savez_compressed(dest / "probabilities.npz", **probabilities, guard_labels=y[guard], outer_labels=target)
    guard_metrics = [bg.metrics(y[guard], g[s], selection["rare"]) for s in supports]
    outer_metrics = [bg.metrics(target, o[s], selection["rare"]) for s in supports]
    nominees = {}
    for method in POLICIES:
        old = slate["shortlists"][f"one_{method}"]
        nominees[method] = supports.index(tuple(old[0])) if old else 0
        idx = nominees[method]
        returned = supports[idx] if outcome(guard_metrics[idx], guard_metrics[0], .03) else anchor
        assert list(returned) == selection["selections"][f"one_{method}"]
    bg.store(dest / "complete.json", dict(protocol=bg.digest(OUT/"protocol.json"), source=selection["shortlists"],
             predictions=bg.digest(dest/"probabilities.npz"), supports=supports, nominees=nominees,
             guard_metrics=guard_metrics, outer_metrics=outer_metrics))


def describe(dataset, fold, record):
    supports = record["supports"]
    guard, outer = record["guard_metrics"], record["outer_metrics"]
    g = np.array([outcome(m, guard[0], .03) for m in guard])
    comparisons, candidates = [], []
    d = 1 - len(supports[-1]) / len(supports[0])
    for i, cols in enumerate(supports):
        candidates.append(dict(dataset=dataset, fold=fold, idx=i, columns=" ".join(map(str, cols)),
                               guard_pass=g[i], **{f"guard_{k}": v for k,v in guard[i].items()},
                               **{f"outer_{k}": v for k,v in outer[i].items()}))
    for tolerance in (.01, .03, .05):
        o = np.array([outcome(m, outer[0], tolerance) for m in outer])
        possible = bool(o[1:].any())
        reachable = bool((g[1:] & o[1:]).any())
        for method, i in record["nominees"].items():
            success = bool(i and g[i] and o[i])
            disposition = ("singleton" if not i else "no_outer_pass" if not possible
                           else "guard_excludes_all" if not reachable else "missed_nomination" if not success
                           else "realized")
            values = dict(no_outer_pass=d*(not possible), guard_excludes_all=d*(possible and not reachable),
                          missed_nomination=d*(reachable and not success), realized=d*success)
            assert abs(sum(values.values()) - d) < 1e-14
            comparisons.append(dict(dataset=dataset,fold=fold,method=method,tolerance=tolerance,
                       candidates=len(supports)-1,nominee=i,disposition=disposition,
                       outer_ceiling=d*possible,joint_ceiling=d*reachable,nominal=d,**values))
    return comparisons, candidates


def report(records):
    EVIDENCE.mkdir(exist_ok=True, parents=True)
    folds, candidates = [], []
    for dataset, fold, record in records:
        f,c = describe(dataset, fold, record)
        folds.extend(f)
        candidates.extend(c)
    frame = pd.DataFrame(folds)
    frame.to_csv(EVIDENCE / "folds.csv", index=False)
    pd.DataFrame(candidates).to_csv(EVIDENCE / "supports.csv", index=False)
    cols = ["nominal","outer_ceiling","joint_ceiling","no_outer_pass","guard_excludes_all","missed_nomination","realized"]
    rows = []
    for (tolerance, method), group in frame.query("dataset != 'optdigits'").groupby(["tolerance", "method"]):
        by_task = group.groupby("dataset")[cols].mean()
        rng = np.random.default_rng(33119)
        draws = rng.integers(len(by_task), size=(30000, len(by_task)))
        for metric in cols:
            differences = by_task[metric].to_numpy()
            lo, hi = np.percentile(differences[draws].mean(axis=1), [2.5, 97.5])
            rows.append(dict(tolerance=tolerance,method=method,metric=metric,mean=differences.mean(),lo95=lo,hi95=hi))
    pd.DataFrame(rows).to_csv(EVIDENCE / "summary.csv", index=False)
    print(frame.query("dataset != 'optdigits' and tolerance == .03").groupby(["method","disposition"]).size())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2)
    args = parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            fit_fold(args.worker[0], int(args.worker[1]))
        return
    OUT.mkdir(exist_ok=True, parents=True)
    dependencies = (DESIGN, Path(__file__), Path(bg.__file__), bg.RUN/"protocol.json", bg.DONOR/"eligibility.csv")
    lock = {p.relative_to(bg.ROOT).as_posix(): bg.digest(p) for p in dependencies}
    protocol = OUT / "protocol.json"
    if protocol.exists() and lock != bg.read(protocol):
        raise ValueError("Opportunity protocol changed")
    bg.store(protocol, lock)
    records = []
    eligibility = pd.read_csv(bg.DONOR / "eligibility.csv")
    jobs = [(r.dataset,r.fold) for r in eligibility.itertuples() if r.eligible] + [("optdigits",0)]
    start = perf_counter()
    for n, (name, fold) in enumerate(jobs, 1):
        dest = OUT/name/f"fold-{fold}"
        if not (dest/"complete.json").exists():
            subprocess.run([sys.executable, "-m", "experiments.context_review.refit_opportunity",
                            "--worker", name, str(fold)], check=True, timeout=3600)
        record = bg.read(dest/"complete.json")
        assert record["protocol"] == bg.digest(protocol) and record["predictions"] == bg.digest(dest/"probabilities.npz")
        records.append((name,fold,record))
        elapsed = perf_counter()-start
        print(f"[{n}/{len(jobs)}] {name}/{fold} elapsed {elapsed:.0f}s, remaining ~{elapsed*(len(jobs)-n)/n:.0f}s", flush=True)
    report(records)


if __name__ == "__main__":
    main()
