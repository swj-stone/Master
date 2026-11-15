"""Independently refit the shared prefix grid and reconstruct each rule's vote."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, recall_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from rounds.pmlb_baselines import DATA, ROOT, RUN, auc, fit, read, sha

HERE = Path(__file__).resolve().parent
GRID = ROOT / "results/context_review/refit-vote-grid"
DONOR = ROOT / "results/context_review/refit-guard"


def measures(y, probabilities, rare):
    prediction = probabilities.argmax(axis=1)
    global_scores = [auc(y, probabilities), accuracy_score(y, prediction)]
    if rare:
        global_scores.append(average_precision_score(y, probabilities[:, 1]))
    return np.array(global_scores), recall_score(y, prediction, average=None)


def within(reference, candidate, tolerance):
    return reference - candidate <= reference * tolerance + 1e-12


def check_fold(name, fold):
    root = GRID / name / f"fold-{fold}"
    original = RUN / name / f"fold-{fold}"
    split = read(original / "selection.json")
    source = read(DONOR / name / f"fold-{fold}" / "proposals.json")
    choice = read(root / "selection.json")
    receipt = read(root / "completed.json")
    table = DATA / name / f"{name}.tsv.gz"
    assert sha(table) == read(original / "completed.json")["data_sha256"]
    assert sha(root / "selection.json") == receipt["selection"]
    assert sha(root / "guard.npz") == choice["guard"]
    assert sha(root / "predictions.npz") == receipt["predictions"]
    raw = pd.read_csv(table, sep="\t")
    y = pd.Categorical(raw.pop("target"), categories=split["class_order"]).codes
    x = raw.to_numpy(dtype=np.float32)
    seed = 20421 + fold
    train, rank, test = [np.asarray(split[key]) for key in ("train", "rank", "test")]
    proposal, guard = train_test_split(rank, test_size=.5, stratify=y[rank], random_state=39823 + fold)
    np.testing.assert_array_equal(guard, source["guard"])
    inner, dev = np.sort(np.r_[train, proposal]), np.sort(np.r_[train, rank])
    k = len(source["candidates"]["anchor"])
    widths = sorted({int(np.ceil(k * fraction)) for fraction in (.25, .5, .75, 1)})
    supports = {w: sorted(source["order"][:w]) for w in widths}
    assert {str(w): cols for w, cols in supports.items()} == choice["supports"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 659]))
    groups = [np.flatnonzero(y[guard] == label) for label in range(len(split["class_order"]))]
    samples = [np.arange(len(guard))]
    for _ in range(3):
        samples.append(np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups]))
    rare = source["rare"]
    tolerances = np.array([.03, .03, .05] if rare else [.03, .03])
    stats = {}
    with np.load(root / "guard.npz") as stored:
        np.testing.assert_array_equal(stored["rows"], guard)
        np.testing.assert_array_equal(stored["panels"], samples)
        transformer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[inner])
        a, b = transformer.transform(x[inner]), transformer.transform(x[guard])
        for w, cols in supports.items():
            p = fit(a[:, cols], y[inner], seed).predict_proba(b[:, cols])
            np.testing.assert_array_equal(p, stored[f"k{w}"])
            stats[w] = [measures(y[guard][ix], p[ix], rare) for ix in samples]
    chosen = {"anchor": k, "global_point": k, "class_point": k, "class_vote": k}
    for w in widths:
        panels = []
        for (global0, recall0), (g, r) in zip(stats[k], stats[w], strict=True):
            panels.append([within(global0, g, tolerances).all(), *within(recall0, r, .03)])
        votes = np.array(panels).T
        np.testing.assert_array_equal(votes, choice["votes"][str(w)])
        for arm, ok in (("global_point", votes[0, 0]), ("class_point", votes[:, 0].all()),
                        ("class_vote", (votes.sum(axis=1) >= 3).all())):
            if ok:
                chosen[arm] = min(chosen[arm], w)
    assert chosen == choice["widths"]
    outcomes = {}
    with np.load(root / "predictions.npz") as stored:
        np.testing.assert_array_equal(stored["rows"], test)
        np.testing.assert_array_equal(stored["y"], y[test])
        transformer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[dev])
        a, b = transformer.transform(x[dev]), transformer.transform(x[test])
        for w in set(chosen.values()):
            cols = supports[w]
            p = fit(a[:, cols], y[dev], seed).predict_proba(b[:, cols])
            np.testing.assert_array_equal(p, stored[f"k{w}"])
            outcomes[w] = measures(y[test], p, rare)
    for row in receipt["rows"]:
        g, r = outcomes[chosen[row["arm"]]]
        np.testing.assert_allclose([row["auc"], row["accuracy"], row["recall"]], [*g[:2], r.mean()], atol=1e-13, rtol=0)
        assert row["exceeds"] == (not within(outcomes[k][0], g, tolerances).all())
        assert row["class_exceeds"] == (not within(outcomes[k][1], r, .03).all())
    return dict(dataset=name, fold=fold, inner_refits=len(widths), final_refits=len(outcomes),
                ballots=len(widths) * len(samples) * (1 + len(groups)))


def main():
    eligible = pd.read_csv(GRID / "eligibility.csv")
    assert len(eligible) == 42 and eligible.eligible.sum() == 41
    with threadpool_limits(limits=1):
        checks = [check_fold(row.dataset, row.fold) for row in eligible.itertuples() if row.eligible]
    frame = pd.DataFrame(checks)
    frame.to_csv(HERE / "refit-review-grid.csv", index=False)
    print(frame[["inner_refits", "final_refits", "ballots"]].sum().to_string())


if __name__ == "__main__":
    main()
