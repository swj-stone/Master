"""Post-hoc, outcome-blind duplicate handling for Spambase's fixed screening rules."""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT, P, ROOT, file_hash, load_dataset, select, train


def evaluate(seed, mask, groups, labels, vectors):
    _, index, frequency = np.unique(groups[mask], return_inverse=True, return_counts=True)
    rows = []
    for (method, learner), (pred, score) in vectors.items():
        for weighting, weights in [('row', None), ('vector', 1.0 / frequency[index])]:
            rows.append(dict(seed=seed, method=method, learner=learner, weighting=weighting,
                             test_rows=len(index), test_vectors=len(frequency),
                             auc=roc_auc_score(labels[mask], score[mask], sample_weight=weights),
                             accuracy=accuracy_score(labels[mask], pred[mask], sample_weight=weights),
                             balanced_accuracy=balanced_accuracy_score(labels[mask], pred[mask], sample_weight=weights)))
    return rows


def run(grouped):
    X, y, task, _, original, _, digest = load_dataset('spambase')
    raw = [hashlib.sha256(row.tobytes()).digest() for row in X]
    _, groups, frequency = np.unique(np.asarray(raw, dtype='V32'), return_inverse=True, return_counts=True)
    contradictory = set(groups[y == 0]) & set(groups[y == 1])
    arm_names = ('exact', 'f', 'full')
    out = OUTPUT / 'summary'
    rows, cases, archive = [], [], {}
    for seed in P['formal_seeds']:
        if grouped:
            splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
            a, b = next(splitter.split(X, y, groups))
            assert not np.intersect1d(groups[a], groups[b]).size
            vectors = {}
            archive[f's{seed}_train_indices'] = a
            archive[f's{seed}_test_indices'] = b
            archive[f's{seed}_truth'] = y[b]
            for method in arm_names:
                chosen, _ = select(method, X[a], y[a], task, seed)
                archive[f's{seed}_{method}_original_features'] = original[chosen]
                for learner in P['learners']:
                    pred, score = train(learner, X[np.ix_(a, chosen)], y[a],
                                        X[np.ix_(b, chosen)], task, seed)
                    vectors[method, learner] = pred, score
                    archive[f's{seed}_{method}_{learner}_pred'] = pred
                    archive[f's{seed}_{method}_{learner}_score'] = score
            use = np.ones(len(b), dtype=bool)
        else:
            with np.load(OUTPUT / 'formal' / 'spambase' / f's{seed}' / 'predictions.npz') as z:
                a, b = z['train_indices'], z['test_indices']
                np.testing.assert_array_equal(z['y_test'], y[b])
                vectors = {(m, l): (z[f'{m}_{l}_pred'], z[f'{m}_{l}_score'])
                           for m in arm_names for l in P['learners']}
            use = ~np.isin(groups[b], groups[a])
        cases.append(dict(seed=seed, all_train_rows=len(a), all_test_rows=len(b), removed_test_rows=int((~use).sum()),
                          evaluated_test_rows=int(use.sum()), evaluated_distinct_vectors=len(np.unique(groups[b][use])),
                          distinct_vectors_with_conflicting_labels=len(contradictory & set(groups[b][use])),
                          evaluated_positive_rows=int(y[b][use].sum())))
        rows.extend(evaluate(seed, use, groups[b], y[b], vectors))
    frame = pd.DataFrame(rows)
    keys = ['seed', 'learner', 'weighting']
    exact = frame[frame.method == 'exact'].set_index(keys)
    comparisons = []
    for reference in ('f', 'full'):
        comparison = frame[frame.method == reference].set_index(keys)
        for metric in ('auc', 'accuracy', 'balanced_accuracy'):
            delta = exact[metric] - comparison[metric]
            for (learner, weighting), group in delta.groupby(['learner', 'weighting']):
                comparisons.append(dict(reference=reference, metric=metric, learner=learner, weighting=weighting,
                                        median=group.median(), minimum=group.min(), maximum=group.max(),
                                        positive=int((group > 1e-12).sum()), tied=int((group.abs() <= 1e-12).sum())))
    family = 'grouped_refit' if grouped else 'filtered_replay'
    result = dict(source_sha256=digest, study_config_sha256=file_hash(ROOT / 'configs/minpi-study.json'),
                  audit_source_sha256=file_hash(ROOT / 'experiments/audit_spambase_duplicates.py'),
                  family=family, total_rows=len(y), distinct_vectors=len(frequency), conflict_groups=len(contradictory),
                  rows_in_conflict_groups=int(np.isin(groups, list(contradictory)).sum()), cases=cases,
                  metrics=rows, paired=comparisons)
    if grouped:
        np.savez_compressed(out / 'spambase_grouped_predictions.npz', **archive)
    (out / f'spambase_{family}.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(family, pd.DataFrame(cases).to_string(index=False))
    print(pd.DataFrame(comparisons).query("reference == 'f' and metric == 'auc'").to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grouped-refit', action='store_true', help='Retrain all method/learner pairs on group-disjoint splits')
    with threadpool_limits(limits=1):
        run(parser.parse_args().grouped_refit)
