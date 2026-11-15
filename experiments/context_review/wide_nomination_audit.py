"""Replay candidate scores from proposal rows with direct weighted positive-negative comparisons."""
import json
from zipfile import ZipFile

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from experiments.context_review.wide_data import EVIDENCE, RUN
from experiments.context_review.wide_score import validate_package
from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import digest
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.piv3_compact_witness import _sample


def direct(y, p, weights, rare):
    acc = (weights * (y == np.argmax(p, axis=1))).sum(1) / weights.sum(1)
    aucs = []
    for col in [1] if p.shape[1] == 2 else range(p.shape[1]):
        keep = y == col
        left, right = p[keep, col, None], p[~keep, col][None, :]
        credit = (left > right).astype(float) + 0.5 * (left == right)
        wins = np.einsum("ip,pn,in->i", weights[:, keep], credit, weights[:, ~keep])
        aucs.append(wins / (weights[:, keep].sum(1) * weights[:, ~keep].sum(1)))
    cols = [acc, np.mean(aucs, axis=0)]
    if rare:
        cols.append([average_precision_score(y, p[:, 1], sample_weight=w) for w in weights])
    return np.array(cols).T


def audit():
    seal = validate_package()
    report = []
    with ZipFile(EVIDENCE / "predictions.zip") as archive, threadpool_limits(limits=1):
        for key in seal["folds"]:
            state = json.loads(archive.read(f"{key}/selection.json"))
            layout = json.loads(archive.read(f"{key}/layout.json"))
            input_file = RUN / key / "inputs.npz"
            assert digest(input_file) == layout["inputs_sha256"]
            with np.load(input_file) as data:
                train, prop = data["train"], data["proposal"]
                raw, labels = data["x"], data["y"]
            point = state["choices"]["point"]
            anchor = frozenset(point["anchor"])
            seed = 20421 + layout["fold"]
            task = "binary" if len(layout["classes"]) == 2 else "multiclass"
            prep = SimpleImputer(strategy="median", keep_empty_features=True).fit(raw[train])
            xt, xp = prep.transform(raw[train]), prep.transform(raw[prop])
            estimator = fit(xt, labels[train], seed)
            gain = estimator.get_booster().get_score(importance_type="gain")
            ranked = sorted((int(k[1:]) for k in gain), key=lambda j: (-gain[f"f{j}"], j))
            assert (ranked or [0]) == point["order"]
            ids, weights = _sample(labels[prop], task, np.random.SeedSequence([seed, 659]))
            y, xx = labels[prop][ids], xp[ids]
            background_idx = np.random.default_rng(seed).choice(len(xt), min(512, len(xt)), replace=False)
            bg = np.median(xt[background_idx], axis=0)
            rare = point["rare"]
            cache = {}

            def score(cols):
                if cols not in cache:
                    columns = sorted(cols)
                    values = np.broadcast_to(bg, xx.shape).copy()
                    values[:, columns] = xx[:, columns]
                    cache[cols] = direct(y, estimator.predict_proba(values), weights, rare)
                return cache[cols]

            family = separating_contexts(point["order"][len(anchor):])
            scale = np.array([.03, .03, .05] if rare else [.03, .03])
            max_error = 0.
            for result in state["pool"]:
                candidate = frozenset(result["columns"])
                margins = []
                for extra in family:
                    reference = score(anchor | extra)
                    rel = scale * abs(reference)
                    signed = score(candidate | extra) - reference + rel + 1e-12
                    margins.append((signed / np.maximum(rel, 1e-12)).min(axis=1))
                margins = np.stack(margins)
                vote = (margins >= 0).sum(axis=1).min()
                slack = np.sort(margins, axis=1)[:, 1].min()
                first = margins[0, 0]
                assert int(vote) == result["vote"]
                difference = np.abs([slack-result["slack"], first-result["point"]]).max()
                max_error = max(max_error, float(difference))
                assert difference <= 1e-8, (key, result, difference)
            for method, choice in state["choices"].items():
                if state["pool"]:
                    assert choice["nominee"] == max(state["pool"], key=lambda rec: rec[method])["columns"]
                else:
                    assert len(anchor) == 1 and choice["nominee"] == sorted(anchor)
            report.append(dict(fold=key, unique_masks=len(cache), candidates=len(state["pool"]),
                               max_score_error=max_error))
    store(EVIDENCE / "nomination-audit.json", report)
    print(f"Replayed {sum(row['candidates'] for row in report)} candidates in {len(report)} folds. "
          f"Maximum absolute continuous-score error {max(row['max_score_error'] for row in report):.3g}")


if __name__ == "__main__":
    audit()
