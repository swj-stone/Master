"""Create the ANOVA exchange slates and seal forest probability vectors before scoring."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Timer
from time import perf_counter
import warnings
from zipfile import ZipFile

import numpy as np
import sklearn
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review.forest_predict import deadline, forest_masker, receiver
from experiments.context_review.receiver_transport import SOURCE, frozen_folds, member


ROOT = bg.ROOT
DESIGN = ROOT / "configs/context_review/filter_neighborhood.json"
DEST = ROOT / "thesis-paper/evidence/filter_neighborhood"
RUN = ROOT / "results/context_review/filter-neighborhood"


def anova_slate(x, y, source_id, fold):
    imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        scores, _ = f_classif(imputed.astype(np.float64), y)
    scores[np.ptp(imputed, axis=0) == 0] = np.nan
    order = np.lexsort((np.arange(x.shape[1]), -np.nan_to_num(scores, nan=-np.inf,
                                                               posinf=np.inf, neginf=-np.inf))).tolist()
    d = len(order)
    width = min(d, max(2, (d + 1) // 2))
    anchor = order[:width]
    if width == 1:
        return order, [sorted(anchor)]
    keep = width - max(1, width // 4)
    prefix, remaining = anchor[:keep], anchor[keep:]
    positions = {col: i for i, col in enumerate(order)}
    count = keep * len(remaining)
    rng = np.random.default_rng(np.random.SeedSequence([87201, source_id, fold]))
    pairs = range(count) if count <= 7 else rng.choice(count, 7, replace=False)
    candidates = [sorted(prefix)]
    for pair in pairs:
        removed, added = divmod(int(pair), len(remaining))
        trial = prefix.copy()
        trial[removed] = remaining[added]
        candidates.append(sorted(trial))
    candidates.sort(key=lambda cols: (sum(positions[c] for c in cols), cols))
    assert candidates[0] == sorted(prefix) and len(set(map(tuple, candidates))) == len(candidates)
    return order, [sorted(anchor), *candidates]


def forecast(panel, key, source_seal, campaign):
    source = ROOT / "results/context_review" / SOURCE[panel] / key
    assert bg.digest(source / "selection.json") == source_seal
    selection = bg.read(source / "selection.json")
    layout = bg.read(source / "layout.json")
    assert bg.digest(source / "layout.json") == selection["layout"]
    assert bg.digest(source / "inputs.npz") == layout["inputs_sha256"]
    with np.load(source / "inputs.npz") as data:
        x, y, outer, train, prop, guard, ids = (data[k] for k in
            ("x", "y", "x_outer", "train", "proposal", "guard", "test_row"))
    seed = 20421 + layout["fold"]
    start = perf_counter()
    order, supports = anova_slate(x[train], y[train], layout["id"], layout["fold"])
    rank_s = perf_counter() - start
    rare = selection["choices"]["gain"]["rare"]
    tensor, names, winners, timings = forest_masker(x, y, train, prop, dict(order=order, rare=rare),
                                                   supports, seed)
    inner = np.sort(np.r_[train, prop])
    np.testing.assert_array_equal(np.sort(np.r_[inner, guard]), np.arange(len(y)))
    guard_probs, guard_times, guard_elapsed = receiver(x, y, inner, x[guard], supports, seed)
    outer_probs, outer_times, outer_elapsed = receiver(x, y, np.arange(len(y)), outer, supports, seed)
    timings.update(ranking_s=rank_s, all_guard_s=guard_elapsed, all_outer_s=outer_elapsed,
                   guard_support_s=guard_times, outer_support_s=outer_times)
    data = BytesIO()
    np.savez_compressed(data, tensor=tensor, train=train, proposal=prop, guard=guard, guard_y=y[guard],
                        test_row=ids, **{f"g{i}": p for i, p in enumerate(guard_probs)},
                        **{f"o{i}": p for i, p in enumerate(outer_probs)})
    receipt = dict(campaign=campaign, panel=panel, dataset=layout["dataset"], fold=layout["fold"],
                   id=layout["id"], source_sha=source_seal, layout_sha=selection["layout"],
                   inputs_sha=layout["inputs_sha256"], targets_sha=layout["targets_sha256"],
                   member_sha=sha256(data.getbuffer()).hexdigest(), metrics=names,
                   order=order, supports=supports, winners=winners, timings=timings)
    return receipt, data.getvalue()


def predict():
    DEST.mkdir(exist_ok=True, parents=True)
    campaign = dict(design_sha=bg.digest(DESIGN), code_sha=bg.digest(Path(__file__)),
                    forest_sha=bg.digest(ROOT / "experiments/context_review/forest_predict.py"),
                    sklearn=sklearn.__version__)
    jobs = frozen_folds()
    records = {}
    start = perf_counter()
    with threadpool_limits(limits=1):
        for i, (panel, key, seal) in enumerate(jobs, 1):
            full = f"{panel}/{key}"
            folder = RUN / full
            folder.mkdir(parents=True, exist_ok=True)
            probs, receipt = folder / "probabilities.npz", folder / "receipt.json"
            before = perf_counter()
            if not receipt.exists():
                timer = Timer(3600, deadline, args=(full,))
                timer.daemon = True
                timer.start()
                try:
                    result, data = forecast(panel, key, seal, campaign)
                    tmp = probs.with_suffix(".tmp")
                    tmp.write_bytes(data)
                    tmp.replace(probs)
                    bg.store(receipt, result)
                finally:
                    timer.cancel()
            rec = bg.read(receipt)
            assert rec["campaign"] == campaign and rec["source_sha"] == seal
            assert rec["member_sha"] == bg.digest(probs)
            records[full] = rec
            elapsed = perf_counter() - start
            print(f"[{i:02}/{len(jobs)}] {full} fold={perf_counter()-before:.1f}s "
                  f"elapsed={elapsed:.1f}s ETA={elapsed*(len(jobs)-i)/i:.1f}s", flush=True)
    temp = DEST / "forecasts.tmp"
    with ZipFile(temp, "w") as archive:
        for name in records:
            member(archive, name+".npz", (RUN / name / "probabilities.npz").read_bytes())
    path = DEST / "predictions.zip"
    temp.replace(path)
    bg.store(DEST / "prediction-lock.json", dict(**campaign, archive_sha=bg.digest(path), folds=records))


if __name__ == "__main__":
    predict()
