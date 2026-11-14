"""Fast, prediction-only permutation importance with three paired shuffles per split-used feature."""
import numpy as np
from scipy.stats import rankdata

from minppi_mofs.baselines._selection_context import Selection
from minppi_mofs.utility import compute_utility


def sampled_rows(labels, limit, regression, seed):
    """Preserve small classes within a capped ranking sample."""
    if len(labels) <= limit:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    if regression:
        return np.sort(rng.choice(len(labels), limit, replace=False))
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) > limit:
        raise ValueError("The row budget is smaller than the number of classes")
    remaining = limit
    blocks = []
    for j, index in enumerate(np.argsort(counts, kind="stable")):
        count = min(int(counts[index]), remaining // (len(classes)-j))
        blocks.append(rng.choice(np.flatnonzero(labels == classes[index]), count, replace=False))
        remaining -= count
    return np.sort(np.concatenate(blocks))


def _batch_score(labels, pred, utility):
    if utility == "r2":
        return np.asarray([compute_utility(labels, p, metric=utility) for p in pred])
    classes = [1] if utility == "roc_auc" else range(pred.shape[-1])
    ranks = rankdata(pred[:, :, classes], axis=1, method="average")
    auc = []
    for j, label in enumerate(classes):
        positive = labels == label
        n = positive.sum()
        auc.append((ranks[:, positive, j].sum(axis=1)-n*(n+1)/2) / (n*(len(labels)-n)))
    return np.mean(auc, axis=0)


def select(context):
    model = context.full_model
    regression = context.task == "regression"
    rows = sampled_rows(context.yr, 128, regression, context.seed)
    y = context.yr[rows]
    x = np.ascontiguousarray(context.ranking.iloc[rows].to_numpy(), dtype=np.float32)
    predict = model.predict if regression else model.predict_proba
    base = _batch_score(y, np.asarray(predict(x), dtype=float)[None], context.utility)[0]
    # A positive split gain identifies coordinates actually used by this reference booster.
    gains = model.get_booster().get_score(importance_type="gain")
    used = sorted(int(name.removeprefix("f")) for name, score in gains.items() if score > 0)
    repeats = 3
    block_size = max(1, min(16, 4_000_000 // (x.size*repeats)))
    scores = np.zeros(len(context.columns))
    for start in range(0, len(used), block_size):
        features = used[start:start+block_size]
        batch = np.broadcast_to(x, (len(features), repeats, *x.shape)).copy()
        for j, feature in enumerate(features):
            rng = np.random.default_rng(np.random.SeedSequence([context.seed, feature]))
            for rep in range(repeats):
                batch[j, rep, :, feature] = x[rng.permutation(len(x)), feature]
        pred = np.asarray(predict(batch.reshape(-1,x.shape[1])), dtype=float)
        values = _batch_score(y,pred.reshape(len(features)*repeats,len(x),*pred.shape[1:]),context.utility)
        scores[features] = base-values.reshape(len(features),repeats).mean(axis=1)
        context.progress("permutation", covered=min(start+block_size,len(used)), total=len(used))
    keep = np.flatnonzero(scores > 0)
    if not len(keep):
        keep = [int(np.argmax(scores))]
    return Selection(scores=scores, selected=[context.columns[i] for i in keep], diagnostics=dict(
        shuffled_features=len(used), repeats=repeats, ranking_rows=rows.tolist(),
        prediction_rows=(1+len(used)*repeats)*len(rows), mean_drops=scores.tolist()))
