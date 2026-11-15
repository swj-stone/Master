"""Adapt a frozen tabular predictor and compact Gain anchor to the context-vote search."""
import numpy as np

from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.baselines._selection_context import Selection
from minppi_mofs.context_vote import search
from minppi_mofs.piv3_compact_tree import select as gain_anchor
from minppi_mofs.piv3_compact_witness import _sample, _within


def select(context, arm):
    task, seed = context.task, context.seed
    model, train, ranking = context.full_model, context.train, context.ranking
    _, info = gain_anchor(model, train, context.yt, ranking, context.yr, task=task,
                          seed=seed, utility=context.utility, aggregate="gain")
    order, k = info["order"], info["k"]
    anchor = frozenset(order[:k])
    tails = ((), order[k:2*k], order[k:])
    names = list(train.columns)
    # All panels, references and backgrounds are frozen before the first proposal.
    labels = np.asarray(context.yr)
    rows, weights = _sample(labels, task, np.random.SeedSequence([seed, 659]))
    positive_counts = np.unique(labels, return_counts=True)[1]
    rare = task == "binary" and positive_counts.min() <= .05 * len(labels)
    score = make_scorer(labels[rows], weights, task, context.utility, rare)
    readout = model.predict if task == "regression" else model.predict_proba
    values = np.ascontiguousarray(ranking.iloc[rows].to_numpy(), dtype=np.float32)
    background = np.random.default_rng(seed).choice(len(train), min(512, len(train)), replace=False)
    fill = np.nanmedian(train.iloc[background].to_numpy(dtype=np.float32), axis=0)
    masked = np.empty_like(values)
    positions = {name:i for i,name in enumerate(names)}
    cache = {}
    def assess(support):
        if support not in cache:
            cols = sorted(positions[c] for c in support)
            masked[:] = fill
            masked[:, cols] = values[:, cols]
            cache[support] = score(readout(masked))
        return cache[support]

    def ballot(subset, extension):
        reference = assess(anchor | extension)
        candidate = assess(subset | extension)
        return tuple(_within(a,b) for a,b in zip(candidate,reference,strict=True))

    live, log = search(order[:k], tails, ballot, rule="min" if arm == "eager" else arm,
                       eager=arm == "eager")
    selected = [name for name in names if name in live]
    info.update(log, arm=arm, anchor_k=k, k=len(selected), witness_predictions=len(cache),
                witness_rows=rows.tolist(), panel_weights=weights.tolist(),
                mask_scores=[dict(mask=sorted(key),scores=scores) for key,scores in cache.items()])
    return Selection(selected=selected, diagnostics=info)
