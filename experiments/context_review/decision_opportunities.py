"""Resolve vote ties uniformly and bound policy contrasts using guarded supports alone."""
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import budgeted_guard as bg
from experiments.context_review.refit_opportunity import outcome


EVIDENCE = bg.ROOT / "thesis-paper/evidence"
RUNS = bg.ROOT / "results/context_review"
DEST = EVIDENCE / "opportunities"
PHASES = {
    "first": ("transport-uci", "low-fidelity", "transport/low-fidelity-lock.json"),
    "wide": ("wide-uci", "wide-slate", "wide_slate/prediction-lock.json"),
}
PAIRS = (("slack", "point"), ("refined", "top_uniform"), ("top_uniform", "vote"),
         ("refined", "vote"), ("top_uniform", "uniform"), ("vote", "gain"))
METRICS = ["u", "breach", "accepted", "saved", "auc", "accuracy"]


def nominees(selection, receipt):
    """Map tying nomination lotteries to explicit candidate indices before the guard."""
    supports = receipt["supports"]
    fixed = {arm: [supports.index(choice["nominee"])] for arm, choice in selection["choices"].items()}
    candidates = selection["pool"]
    if not candidates:
        assert len(supports[0]) == 1 and len(supports) == 1
        return {**fixed, "top_uniform": [0], "uniform": [0], "refined": [0]}
    all_ids = [supports.index(c["columns"]) for c in candidates]
    assert sorted(all_ids) == list(range(1, len(supports)))
    best_vote = max(c["vote"] for c in candidates)
    top = [(idx, c) for idx, c in zip(all_ids, candidates) if c["vote"] == best_vote]
    assert fixed["vote"] == [top[0][0]]  # the stored tie order is Gain then sorted column IDs
    refined = max(top, key=lambda x: x[1]["point"])[0]
    return {**fixed, "top_uniform": [i for i, _ in top], "uniform": all_ids, "refined": [refined]}


def guarded_distribution(indices, passes, n):
    returned = [i if passes[i] else 0 for i in indices]
    return np.bincount(returned, minlength=n) / len(indices)


def envelope(a, b, savings):
    # A common support has a single passage status under either policy. Any
    # shared probability on it cancels before we bound that status by [0,1].
    coefficients = (a - b) * savings
    return np.minimum(coefficients, 0).sum(), np.maximum(coefficients, 0).sum()


def measures(y, probs, prefix, n, rare):
    return [bg.metrics(y, probs[f"{prefix}{i}"], rare) for i in range(n)]


def fold_records(panel, key, rec, labels, logit_archive, expected_receipt):
    base_folder, slate_folder, _ = PHASES[panel]
    source = RUNS / base_folder / key
    complete = RUNS / slate_folder / key
    assert bg.digest(source / "selection.json") == rec["selection_sha"]
    assert bg.digest(complete / "sealed.json") == expected_receipt
    slate = bg.read(complete / "sealed.json")
    assert bg.digest(complete / "predictions.npz") == slate["predictions"]
    assert slate["supports"] == rec["supports"] and slate["rare"] == rec["rare"]
    rare = rec["rare"]
    selection = bg.read(source / "selection.json")
    choices = nominees(selection, slate)
    size = len(slate["supports"])
    for arm, i in rec["nominees"].items():
        assert choices[arm] == [i]
    savings = np.array([1 - len(c) / len(slate["supports"][0]) for c in slate["supports"]])
    ident = {k: rec[k] for k in ("panel", "dataset", "fold")}
    logit_bytes = logit_archive.read(f"{panel}/{key}.npz")
    assert sha256(logit_bytes).hexdigest() == rec["predictions_sha"]
    bounds, rows = [], []
    with np.load(BytesIO(logit_bytes)) as logistic, np.load(complete / "predictions.npz") as xgb:
        guard = {"logistic": measures(logistic["guard_y"], logistic, "g", size, rare),
                 "xgb": slate["guard_metrics"]}
        final = {}
        for receiver, scores in guard.items():
            passes = [outcome(v, scores[0], .03) for v in scores]
            final[receiver] = {arm: guarded_distribution(ids, passes, size) for arm, ids in choices.items()}
            for a, b in PAIRS:
                pa, pb = final[receiver][a], final[receiver][b]
                lo, hi = envelope(pa, pb, savings)
                bounds.append(dict(**ident, receiver=receiver, policy=a, reference=b,
                                   lower=lo, upper=hi, distinct=bool(np.any(pa != pb))))
        # The envelope above depends on proposals and guards only. Labels are
        # opened here to score each candidate and the lotteries retrospectively.
        truth_bytes = labels.read(f"{panel}/{key}.npz")
        assert sha256(truth_bytes).hexdigest() == rec["targets_sha"]
        with np.load(BytesIO(truth_bytes)) as truth:
            np.testing.assert_array_equal(truth["test_row"], logistic["test_row"])
            for receiver, pred in (("xgb", xgb), ("logistic", logistic)):
                metrics = measures(truth["y"], pred, "o", size, rare)
                ok = np.array([outcome(v, metrics[0], .03) for v in metrics])
                vals = pd.DataFrame(dict(u=savings * ok, breach=(~ok).astype(int),
                                         accepted=(savings > 0).astype(int), saved=savings,
                                         auc=[m["auc"] for m in metrics], accuracy=[m["accuracy"] for m in metrics]))
                top = final[receiver]["top_uniform"] > 0
                mixed = bool(vals.u[top].nunique() > 1)
                passes = [outcome(v, guard[receiver][0], .03) for v in guard[receiver]]
                joint_ids = [i for i in choices["top_uniform"] if i and passes[i] and ok[i]]
                for arm, prob in final[receiver].items():
                    scores = prob @ vals.to_numpy()
                    rows.append(dict(**ident, receiver=receiver, method=arm, candidates=size-1,
                                     top_count=len(choices["top_uniform"]) if size > 1 else 0,
                                     top_joint_pass_count=len(joint_ids), mixed_top=mixed,
                                     nominated_ids=" ".join(map(str, choices[arm])),
                                     returned_ids=" ".join(map(str, np.flatnonzero(prob))),
                                     **dict(zip(METRICS, scores))))
    return bounds, rows


def summarize(folds, bounds):
    keys = ["panel", "receiver"]
    tasks = folds.groupby([*keys, "method", "dataset"])[METRICS].mean()
    means = tasks.groupby([*keys, "method"]).mean()
    estimates = []
    for (panel, receiver), part in tasks.groupby(keys):
        group = part.droplevel(keys)
        for a, b in PAIRS:
            diffs = group.loc[a] - group.loc[b]
            n = len(diffs)
            draws = np.random.default_rng(33119).integers(n, size=(30000, n))
            for field in METRICS:
                delta = diffs[field].to_numpy()
                lower, upper = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
                estimates.append(dict(panel=panel, receiver=receiver, policy=a, reference=b, metric=field,
                                      mean=delta.mean(), low95=lower, high95=upper))

    base = [*keys, "policy", "reference"]
    task_bounds = bounds.groupby([*base, "dataset"])[["lower", "upper"]].mean()
    summary = []
    for key, part in task_bounds.groupby(base):
        n = len(part)
        positive = int((part.upper > 0).sum())
        summary.append(dict(**dict(zip(base, key)), lower=part.lower.mean(), upper=part.upper.mean(),
                            tasks=n, potentially_positive_tasks=positive,
                            chance_bootstrap_omits_potential_gains=((n-positive) / n) ** n))
    return tasks, means, pd.DataFrame(estimates), pd.DataFrame(summary)


def main():
    lock = bg.read(EVIDENCE / "receiver/prediction-lock.json")
    assert len(lock["folds"]) == 46
    archive = EVIDENCE / "receiver/predictions.zip"
    assert bg.digest(archive) == lock["forecast_archive_sha"]
    receipts = {p: bg.read(EVIDENCE / fields[2]) for p, fields in PHASES.items()}
    assert set(lock["folds"]) == {f"{p}/{k}" for p, seals in receipts.items() for k in seals}
    bound_rows, policy_rows = [], []
    with ZipFile(archive) as logistic, ZipFile(EVIDENCE / "receiver/outcomes.zip") as labels:
        for full_key, rec in lock["folds"].items():
            panel, key = full_key.split("/", 1)
            b, p = fold_records(panel, key, rec, labels, logistic, receipts[panel][key])
            bound_rows.extend(b)
            policy_rows.extend(p)
    bounds, folds = pd.DataFrame(bound_rows), pd.DataFrame(policy_rows)
    tasks, means, contrasts, envelopes = summarize(folds, bounds)
    # Every delivered contrast must lie within its label-free fold envelope.
    observed = folds.pivot(index=["panel", "receiver", "dataset", "fold"], columns="method", values="u")
    for row in bounds.itertuples():
        saved = observed.loc[(row.panel, row.receiver, row.dataset, row.fold)]
        assert row.lower-1e-12 <= saved[row.policy]-saved[row.reference] <= row.upper+1e-12
    DEST.mkdir(parents=True, exist_ok=True)
    for filename, frame in (("folds", folds), ("tasks", tasks.reset_index()), ("means", means.reset_index()),
                            ("contrasts", contrasts), ("fold-envelopes", bounds), ("envelopes", envelopes)):
        frame.to_csv(DEST / f"{filename}.csv", index=False)


if __name__ == "__main__":
    main()
