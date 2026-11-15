"""Independently replay frozen candidate scores with scikit-learn reference metrics."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review.transport_data import EVIDENCE, RUN, SPEC, read
from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import digest
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.piv3_compact_witness import _sample, _measure


def audit():
    seal = read(EVIDENCE / "prediction-lock.json")
    largest = 0.
    folds = pairs = masks = 0
    tied_under_roundoff = []
    for name, *_ in read(SPEC)["sources"]:
        for fold in range(3):
            key = f"{name}/fold-{fold}"
            path = RUN / key
            assert digest(path / "selection.json") == seal["selections"][key]
            info = read(path / "selection.json")
            folds += 1
            if not info["pool"]:
                continue
            state = info["choices"]["gain"]
            anchor = frozenset(state["anchor"])
            order = state["order"]
            seed = 20421+fold
            with np.load(path / "inputs.npz") as data:
                x, y = data["x"], data["y"]
                train, prop = data["train"], data["proposal"]
                impute = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
                xt, xp = impute.transform(x[train]), impute.transform(x[prop])
                model = fit(xt, y[train], seed)
                task = "binary" if model.n_classes_ == 2 else "multiclass"
                ids, weights = _sample(y[prop], task, np.random.SeedSequence([seed,659]))
                labels = y[prop][ids]
            rng = np.random.default_rng(seed)
            bg = np.median(xt[rng.choice(len(xt), min(512, len(xt)), replace=False)], axis=0)
            family = separating_contexts(order[len(anchor):])
            cache = {}

            def reference(mask):
                if mask not in cache:
                    cols = sorted(mask)
                    buffer = np.broadcast_to(bg, xp[ids].shape).copy()
                    buffer[:, cols] = xp[ids][:, cols]
                    probability = model.predict_proba(buffer)
                    cache[mask] = [_measure(labels, probability, weight, task, state["rare"])
                                   for weight in weights]
                return cache[mask]

            repl = []
            for candidate in info["pool"]:
                stats = []
                for extension in family:
                    supports = frozenset(candidate["columns"]) | extension, anchor | extension
                    panels = []
                    for a, b in zip(*(reference(s) for s in supports), strict=True):
                        residuals = []
                        for metric, value in b.items():
                            limit = (.05 if metric == "ap" else .03)*abs(value)
                            residuals.append((limit-(value-a[metric])+1e-12)/max(limit,1e-12))
                        panels.append(min(residuals))
                    stats.append(panels)
                numbers = np.array(stats)
                vote = int(np.sum(numbers>=0, axis=1).min())
                slack = float(np.sort(numbers, axis=1)[:,1].min())
                point = float(numbers[0,0])
                assert vote == candidate["vote"]
                error = max(abs(candidate["slack"]-slack), abs(candidate["point"]-point))
                assert error < 1e-8, (name, fold, candidate["columns"], error)
                largest = max(largest,error)
                pairs += 1
                repl.append(dict(columns=candidate["columns"], vote=vote, slack=slack, point=point))
            for arm in ("vote","slack","point"):
                comparison = max(repl, key=lambda c:c[arm])
                selected = info["choices"][arm]["nominee"]
                if comparison["columns"] != selected:
                    chosen = next(c for c in repl if c["columns"] == selected)
                    assert abs(chosen[arm]-comparison[arm]) < 1e-8
                    tied_under_roundoff.append(dict(dataset=name, fold=fold, arm=arm))
            masks += len(cache)
        print(f"{name}: cumulative {pairs} supports, {masks} distinct masks", flush=True)
    store(EVIDENCE / "reference-metric-audit.json",
          dict(folds=folds, candidates=pairs, distinct_masks=masks, maximum_absolute_error=largest,
               near_tie_reversals=tied_under_roundoff,
               prediction_lock_sha256=digest(EVIDENCE / "prediction-lock.json")))
    print("Max absolute score error:", largest, "near-tie reversals:", tied_under_roundoff)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        audit()
