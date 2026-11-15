"""Nominate half-width sets using common joint-row backgrounds from the fitting split."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import transport_predict as primary
from experiments.context_review.receiver_transport import EVIDENCE, ROOT, SOURCE, frozen_folds
from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.piv3_compact_witness import _sample


DESIGN = ROOT / "configs/context_review/marginal_background.json"
DEST = EVIDENCE / "marginal"
RUN = ROOT / "results/context_review/marginal-background"
REPLICATES = ("A", "B")


def averaged_predictions(model, rows, donors, visible):
    """Average row-matched class probabilities over the same empirical donor distribution."""
    columns = sorted(visible)
    # Each block holds all donors for one or more proposal rows.
    block_rows = max(1, 32 * 1024**2 // (donors.nbytes))
    result = np.empty((len(rows), model.n_classes_), dtype=np.float64)
    for start in range(0, len(rows), block_rows):
        block = rows[start:start + block_rows]
        masked = np.broadcast_to(donors, (len(block), *donors.shape)).copy()
        masked[:, :, columns] = block[:, None, columns]
        probabilities = model.predict_proba(masked.reshape(-1, donors.shape[1]))
        result[start:start + len(block)] = probabilities.reshape(len(block), len(donors), -1).mean(axis=1, dtype=np.float64)
    return result


class MarginalBallot:
    def __init__(self, model, rows, donors, anchor, scorer, deadline):
        self.model, self.rows, self.donors = model, rows, donors
        self.anchor, self.scorer, self.deadline = frozenset(anchor), scorer, deadline
        self.scores = {}

    def score(self, support):
        if support not in self.scores:
            if perf_counter() > self.deadline:
                raise TimeoutError("The 45-minute donor-replicate limit was reached")
            probabilities = averaged_predictions(self.model, self.rows, self.donors, support)
            self.scores[support] = self.scorer(probabilities)
        return self.scores[support]


def nominate_fold(panel, key, selection_sha):
    src = ROOT / "results/context_review" / SOURCE[panel] / key
    saved = bg.read(src / "selection.json")
    layout = bg.read(src / "layout.json")
    assert bg.digest(src / "selection.json") == selection_sha
    assert bg.digest(src / "layout.json") == saved["layout"]
    assert bg.digest(src / "inputs.npz") == layout["inputs_sha256"]
    anchor = saved["choices"]["gain"]["anchor"]
    pool = [c["columns"] for c in saved["pool"]]
    order = saved["choices"]["gain"]["order"]
    template = dict(dataset=layout["dataset"], fold=layout["fold"], panel=panel, selection=selection_sha,
                    supports=[anchor] + pool, design=bg.digest(DESIGN), implementation=bg.digest(Path(__file__)))
    controls = {method: template["supports"].index(saved["choices"][method]["nominee"]) for method in ("point", "slack")}
    template["median_nominees"] = controls
    if not pool:
        return dict(**template, fit_seconds=0., replicates={rep: {} for rep in REPLICATES})

    started = perf_counter()
    with np.load(src / "inputs.npz") as archive:
        x, y, train, prop = (archive[k] for k in ("x", "y", "train", "proposal"))
    transform = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    xt, xp = transform.transform(x[train]), transform.transform(x[prop])
    fold_seed = 20421 + layout["fold"]
    model = bg.fit(xt, y[train], fold_seed)
    fit_seconds = perf_counter() - started
    task = "binary" if model.n_classes_ == 2 else "multiclass"
    rare = saved["choices"]["gain"]["rare"]
    ids, weights = _sample(y[prop], task, np.random.SeedSequence([fold_seed, 659]))
    rows, labels = xp[ids], y[prop][ids]
    families = {"point": [frozenset()], "slack": primary.separating_contexts(order[len(anchor):])}

    # Check the original fitted learner, candidate IDs, panels and metric code
    # against the archived scores for every candidate before changing the replacement.
    median = primary.build_ballot(model, xt, xp, y[prop], fold_seed, anchor, rare)
    for method, family in families.items():
        scores = [primary.rating(median, cols, family, method) for cols in pool]
        np.testing.assert_allclose(scores, [c[method] for c in saved["pool"]], atol=1e-10, rtol=1e-10)
    records = {}
    for rep, name in enumerate(REPLICATES):
        deadline = perf_counter() + 45*60
        seed = np.random.SeedSequence([51731, int(panel == "wide"), layout["id"], layout["fold"], rep])
        donor_ids = np.random.default_rng(seed).choice(len(xt), 16, replace=False)
        donors = xt[donor_ids]
        # Direct per-donor implementation on held proposal rows tests the batch layout.
        check_cols = pool[0]
        expected = []
        for donor in donors:
            buffer = np.broadcast_to(donor, (min(3, len(rows)), len(donor))).copy()
            buffer[:, check_cols] = rows[:len(buffer), check_cols]
            expected.append(model.predict_proba(buffer))
        np.testing.assert_allclose(averaged_predictions(model, rows[:3], donors, check_cols),
                                   np.mean(expected, axis=0, dtype=np.float64), atol=1e-12, rtol=0)
        record = dict(donor_training_positions=donor_ids.tolist(), donor_development_rows=train[donor_ids].tolist())
        for method, family in families.items():
            start = perf_counter()
            scorer = make_scorer(labels, weights[:1] if method == "point" else weights, task, "ovr_auc", rare)
            ballot = MarginalBallot(model, rows, donors, anchor, scorer, deadline)
            scores = [primary.rating(ballot, cols, family, method) for cols in pool]
            assert np.isfinite(scores).all()
            record[method] = dict(nominee=1+int(np.argmax(scores)), scores=scores, masks=len(ballot.scores),
                                  model_rows=len(rows)*len(donors)*len(ballot.scores), seconds=perf_counter()-start)
        records[name] = record
    return dict(**template, fit_seconds=fit_seconds, replicates=records)


def lock():
    DEST.mkdir(exist_ok=True, parents=True)
    RUN.mkdir(exist_ok=True, parents=True)
    records = {}
    started = perf_counter()
    with threadpool_limits(limits=1):
        folds = frozen_folds()
        for j, (panel, key, selection_sha) in enumerate(folds, 1):
            name = f"{panel}/{key}"
            path = RUN / name / "nomination.json"
            if not path.exists():
                path.parent.mkdir(exist_ok=True, parents=True)
                record = nominate_fold(panel, key, selection_sha)
                bg.store(path, record)
            record = bg.read(path)
            assert record["selection"] == selection_sha and record["design"] == bg.digest(DESIGN)
            assert record["implementation"] == bg.digest(Path(__file__))
            records[name] = record
            elapsed = perf_counter() - started
            print(f"[{j}/{len(folds)}] {name}, {elapsed:.1f}s elapsed, {elapsed*(len(folds)-j)/j:.1f}s estimated remaining", flush=True)
    assert len(records) == 46 and sum(len(v["supports"])-1 for v in records.values()) == 216
    bg.store(DEST / "nomination-lock.json", dict(design=bg.digest(DESIGN), implementation=bg.digest(Path(__file__)), folds=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    lock()
