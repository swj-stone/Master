"""External within-fold ranking of passing versus failing equal-width candidates."""
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.context_review import budgeted_guard as bg
from experiments.context_review.decision_opportunities import EVIDENCE, PHASES, RUNS, measures
from experiments.context_review.refit_opportunity import outcome


DEST = EVIDENCE / "concordance"
SCORES = ("point", "slack", "vote", "gain_order", "gain_rank_sum")
TARGETS = ("outer_pass", "joint_pass")
CONTRASTS = (("point", "vote"), ("slack", "vote"), ("slack", "point"), ("point", "gain_order"))


def collect_candidates():
    records = bg.read(EVIDENCE / "receiver/prediction-lock.json")["folds"]
    receipts = {p: bg.read(EVIDENCE / info[2]) for p, info in PHASES.items()}
    assert set(records) == {f"{p}/{k}" for p, entries in receipts.items() for k in entries}
    logistic = pd.read_csv(EVIDENCE / "receiver/candidates.csv").set_index(["panel", "dataset", "fold", "index"])
    candidates, layout = [], []
    with ZipFile(EVIDENCE / "receiver/outcomes.zip") as labels:
        for name, rec in records.items():
            panel, key = name.split("/", 1)
            ident = {k: rec[k] for k in ("panel", "dataset", "fold")}
            base_dir, slate_dir, _ = PHASES[panel]
            source = RUNS / base_dir / key / "selection.json"
            slate_path = RUNS / slate_dir / key / "sealed.json"
            assert bg.digest(source) == rec["selection_sha"] and bg.digest(slate_path) == receipts[panel][key]
            selection, slate = bg.read(source), bg.read(slate_path)
            supports = slate["supports"]
            assert supports == rec["supports"] and len(selection["pool"]) == len(supports) - 1
            layout.append(dict(**ident, candidates=len(supports)-1))
            raw = labels.read(f"{name}.npz")
            assert sha256(raw).hexdigest() == rec["targets_sha"]
            probs = slate_path.with_name("predictions.npz")
            assert bg.digest(probs) == slate["predictions"]
            with np.load(BytesIO(raw)) as truth, np.load(probs) as pred:
                outer = measures(truth["y"], pred, "o", len(supports), rec["rare"])
            guard = slate["guard_metrics"]
            positions = {c: i for i, c in enumerate(selection["choices"]["gain"]["order"])}
            ordered = [tuple(row["columns"]) for row in selection["pool"]]
            assert ordered == sorted(ordered, key=lambda s: (sum(positions[c] for c in s), s))
            assert set(ordered) == set(map(tuple, supports[1:])) and len(set(ordered)) == len(ordered)
            for order, row in enumerate(selection["pool"]):
                cols = row["columns"]
                i = supports.index(cols)
                fit = logistic.loc[panel, rec["dataset"], rec["fold"], i]
                assert tuple(map(int, str(fit["columns"]).split())) == tuple(cols)
                scores = {k: row[k] for k in ("vote", "slack", "point")}
                scores.update(gain_order=-order, gain_rank_sum=-sum(positions[c] for c in cols))
                assert row["gain"] == int(order == 0)
                for receiver, g, o in (("xgb", outcome(guard[i], guard[0], .03), outcome(outer[i], outer[0], .03)),
                                       ("logistic", fit.guard_ok, fit.outer_ok)):
                    candidates.append(dict(**ident, receiver=receiver, candidate_id=i, columns=" ".join(map(str, cols)),
                                           anchor_width=len(supports[0]), width=len(cols), **scores,
                                           guard_pass=bool(g), outer_pass=bool(o), joint_pass=bool(g and o)))
    return pd.DataFrame(candidates), pd.DataFrame(layout)


def fold_concordance(candidates, layout):
    rows = []
    groups = candidates.groupby(["panel", "receiver", "dataset", "fold"])
    for f in layout.itertuples(index=False):
        for receiver in ("xgb", "logistic"):
            key = (f.panel, receiver, f.dataset, f.fold)
            data = groups.get_group(key) if f.candidates else candidates.iloc[:0]
            assert len(data) == f.candidates
            for target in TARGETS:
                passed = data[target].to_numpy(dtype=bool)
                pairs = int(passed.sum() * (~passed).sum())
                for score in SCORES:
                    values = data[score].to_numpy()
                    differences = values[passed, None] - values[~passed][None, :]
                    wins = np.count_nonzero(differences > 0)
                    ties = np.count_nonzero(differences == 0)
                    value = (wins + 0.5 * ties) / pairs if pairs else np.nan
                    if pairs:
                        np.testing.assert_allclose(value, roc_auc_score(passed, values), atol=1e-12, rtol=0)
                    rows.append(dict(panel=f.panel, receiver=receiver, dataset=f.dataset, fold=f.fold,
                                     target=target, score=score, candidates=len(data), passing=passed.sum(),
                                     pairs=pairs, wins=wins, ties=ties, concordance=value))
    return pd.DataFrame(rows)


def summarize(folds):
    groups = ["panel", "receiver", "target", "score"]
    tasks = folds.groupby([*groups, "dataset"], as_index=False).agg(
        concordance=("concordance", "mean"), mixed_folds=("concordance", "count"), folds=("fold", "size"),
        eligible_folds=("candidates", lambda s: (s > 0).sum()), pairs=("pairs", "sum"))
    summary = tasks.groupby(groups, as_index=False).agg(
        concordance=("concordance", "mean"), contributing_tasks=("concordance", "count"), tasks=("dataset", "size"),
        mixed_folds=("mixed_folds", "sum"), eligible_folds=("eligible_folds", "sum"), folds=("folds", "sum"),
        pairs=("pairs", "sum"), min_task=("concordance", "min"), max_task=("concordance", "max"))
    contrasts = []
    for (panel, receiver, target), group in tasks.groupby(groups[:3]):
        values = group.pivot(index="dataset", columns="score", values="concordance").dropna(how="all")
        assert values.notna().all().all()
        n = len(values)
        draws = np.random.default_rng(33119).integers(n, size=(30000, n))
        for a, b in CONTRASTS:
            delta = (values[a] - values[b]).to_numpy()
            lo, hi = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
            contrasts.append(dict(panel=panel, receiver=receiver, target=target, score=a, reference=b,
                                  contributing_tasks=n, mean=delta.mean(), low95=lo, high95=hi))
    return tasks, summary, pd.DataFrame(contrasts)


def main():
    candidates, layout = collect_candidates()
    assert len(layout) == 46 and len(candidates) == 432
    folds = fold_concordance(candidates, layout)
    tasks, summary, contrasts = summarize(folds)
    DEST.mkdir(parents=True, exist_ok=True)
    for name, frame in (("candidates", candidates), ("folds", folds), ("tasks", tasks),
                        ("summary", summary), ("contrasts", contrasts)):
        frame.to_csv(DEST / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
