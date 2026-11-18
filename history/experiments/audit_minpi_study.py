"""Replay the frozen study's predictions, split identities, supports and decision rules."""

import hashlib
import json
from itertools import combinations
from math import ceil

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

from experiments.minpi_study import OUTPUT, P, ROOT, file_hash, load_dataset


def same(left, right):
    np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-12, equal_nan=True)


def binary_auc(labels, score):
    positive = np.count_nonzero(labels)
    ranks = rankdata(score)
    return (ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * (len(labels) - positive))


def replay_metrics(labels, pred, score, learner):
    if score is None:
        error = labels - pred
        return dict(r2=1 - error @ error / np.square(labels - labels.mean()).sum(),
                    mae=np.abs(error).mean())
    unique = np.unique(labels)
    np.testing.assert_array_equal(unique, np.arange(len(unique)))
    if score.ndim == 1:
        expected = score > (.5 if learner == 'trees' else 0)
        auc = binary_auc(labels == 1, score)
    else:
        expected = score.argmax(axis=1)
        auc = np.mean([binary_auc(labels == i, score[:, i]) for i in unique])
    np.testing.assert_array_equal(pred, expected)
    correct = pred == labels
    return dict(auc=auc, accuracy=correct.mean(),
                balanced_accuracy=np.mean([correct[labels == i].mean() for i in unique]))


def quality(candidate, reference, classification):
    tolerance = 1e-12
    if classification:
        return ((candidate.auc - reference.auc >= -.02 - tolerance) &
                (candidate.accuracy - reference.accuracy >= -.03 - tolerance))
    return candidate.r2 - reference.r2 >= -.03 - tolerance


def paired_rules(frame, pair_table):
    decisions = []
    for (name, learner), group in frame.groupby(['dataset', 'learner']):
        group = group.set_index(['arm', 'seed']).sort_index()
        classification = 'auc' in group.loc['full'].dropna(axis=1)
        metric = 'auc' if classification else 'r2'
        for method, reference in ([('exact', arm) for arm in P['arms'] if arm != 'exact'] +
                                  [('exact_forest', 'forest')]):
            a, b, full = (group.loc[label] for label in (method, reference, 'full'))
            relative = quality(a, b, classification)
            absolute = quality(a, full, classification)
            ratio = a.total_seconds / b.total_seconds
            width_ratio = a.k / b.k
            speed = (ratio <= .8) & (width_ratio <= 1)
            compact = (width_ratio <= .8) & (ratio <= 1)
            row = pair_table.loc[(name, learner, method, reference)]
            same((a[metric] - b[metric]).median(), row.primary_difference_median)
            same(ratio.median(), row.time_ratio_median)
            # Summary used exact binary-float comparisons at inclusive decimal margins.
            # Log a changed count only when an observed delta is within roundoff of a boundary.
            difference = int(relative.sum()) - int(row.quality_pass_splits)
            speed_difference = int((relative & speed).sum()) - int(row.quality_and_speed_pass_splits)
            at_boundary = np.isclose(a[metric] - b[metric], -.02 if classification else -.03,
                                     rtol=0, atol=1e-12)
            if classification:
                at_boundary |= np.isclose(a.accuracy - b.accuracy, -.03, rtol=0, atol=1e-12)
            assert abs(difference) <= at_boundary.sum() and abs(speed_difference) <= (at_boundary & speed).sum()
            decisions.append(dict(dataset=name, learner=learner, method=method, reference=reference,
                median_time_ratio=float(ratio.median()), median_width_ratio=float(width_ratio.median()),
                named_quality=int(relative.sum()), full_quality=int(absolute.sum()),
                named_fast=int((relative & speed).sum()), named_quality_count_roundoff_delta=difference,
                named_speed_count_roundoff_delta=speed_difference,
                joint_fast_seeds=a.index[relative & absolute & speed].tolist(),
                joint_compact_seeds=a.index[relative & absolute & compact].tolist()))
    return decisions


def audit_chain():
    from experiments.minpi_study_chain import ARMS, DATASETS, OUT, dataset, source_hashes

    table = pd.read_csv(OUT / 'metrics.csv').set_index(['dataset', 'seed', 'arm', 'learner'])
    paired = pd.read_csv(OUT / 'paired.csv').set_index(['dataset', 'seed', 'learner', 'arm', 'comparator'])
    assert table.index.is_unique and paired.index.is_unique
    expected_hashes = source_hashes()
    vectors = scalars = checked_pairs = copies = 0
    joint = []
    duplicates = {}
    cnae_filtered = []
    for name in DATASETS:
        X, y, task, order, digest, times = dataset(name)
        p = X.shape[1]
        hashes = [hashlib.sha256(row.tobytes()).digest() for row in X] if name in ('secom', 'cnae') else []
        del X
        for seed in P['formal_seeds']:
            folder = OUT / name / f's{seed}'
            rec = json.loads((folder / 'record.json').read_text(encoding='utf-8'))
            assert rec['source_sha256'] == expected_hashes and rec['source'] == digest
            if times is None:
                a, b = train_test_split(np.arange(len(y)), test_size=.25, random_state=seed,
                                        stratify=y if task == 'classification' else None)
            else:
                ordered = np.argsort(times, kind='stable')
                boundary = 3 * len(y) // 4
                a, b = ordered[:boundary], ordered[boundary:]
                assert times[a].max() < times[b].min() and y[b].sum() == rec['failures'] == 24
            if hashes:
                train_vectors = {hashes[i] for i in a}
                unseen = np.asarray([hashes[i] not in train_vectors for i in b])
                duplicates.setdefault(name, {})[seed] = int((~unseen).sum())
            with np.load(folder / 'predictions.npz') as z:
                np.testing.assert_array_equal(z['test_indices'], b)
                np.testing.assert_array_equal(z['train_indices'], a)
                np.testing.assert_array_equal(z['y_test'], y[b])
                np.testing.assert_array_equal(z['original_column_indices'], order)
                if name == 'cnae':
                    for learner in P['learners']:
                        values = {}
                        for arm in ('exact', 'f'):
                            values[arm] = replay_metrics(y[b][unseen], z[f'{arm}_{learner}_pred'][unseen],
                                                         z[f'{arm}_{learner}_score'][unseen], learner)
                        cnae_filtered.append(dict(seed=seed, learner=learner, surviving_test_rows=int(unseen.sum()),
                            **{f'delta_{k}': v - values['f'][k] for k, v in values['exact'].items()}))
                for arm in ARMS:
                    screen = rec['arms'][arm]['screen_indices']
                    selected = rec['arms'][arm]['selected_indices']
                    assert len(set(screen)) == len(screen) == (p if arm == 'none' else ceil(.4 * p))
                    assert len(set(selected)) == len(selected) == ceil(.1 * p)
                    assert set(selected) <= set(screen) <= set(range(p))
                    for learner in P['learners']:
                        row = table.loc[(name, seed, arm, learner)]
                        pred = z[f'{arm}_{learner}_pred']
                        score = z[f'{arm}_{learner}_score'] if task == 'classification' else None
                        measured = replay_metrics(y[b], pred, score, learner)
                        if name == 'secom':
                            idx = np.argsort(-score, kind='stable')
                            hits = np.cumsum(y[b][idx])
                            ends = np.r_[np.flatnonzero(np.diff(score[idx])), len(y[b]) - 1]
                            positive = np.diff(hits[ends], prepend=0)
                            measured['ap'] = np.sum(positive * hits[ends] / (ends + 1)) / hits[-1]
                        for key, value in measured.items():
                            same(value, row[key])
                        same(row.total_s, row.imputation_s + row.screen_s + row.selector_s + row.predict_fit_s)
                        vectors += 1
                        scalars += len(measured)
                formal = OUTPUT / 'formal' / name / f's{seed}'
                if formal.exists():
                    full = json.loads((formal / 'record.json').read_text(encoding='utf-8'))['arms']['full']
                    with np.load(formal / 'predictions.npz') as earlier:
                        for arm, old in [('none', 'forest'), ('exact', 'exact_forest')]:
                            for learner in P['learners']:
                                np.testing.assert_array_equal(z[f'{arm}_{learner}_pred'], earlier[f'{old}_{learner}_pred'])
                                if task == 'classification':
                                    np.testing.assert_array_equal(z[f'{arm}_{learner}_score'], earlier[f'{old}_{learner}_score'])
                                copies += 1
                else:
                    full = None
            for learner in P['learners']:
                exact = table.loc[(name, seed, 'exact', learner)]
                for baseline in ('none', 'f', 'random'):
                    rival = table.loc[(name, seed, baseline, learner)]
                    pair = paired.loc[(name, seed, learner, 'exact', baseline)]
                    for key in ('auc', 'accuracy', 'r2', 'mae', 'ap', 'balanced_accuracy'):
                        same(exact[key] - rival[key], pair[f'delta_{key}'])
                    ratio = exact.total_s / rival.total_s
                    same(ratio, pair.time_ratio)
                    named = quality(exact, rival, task == 'classification')
                    joint.append(dict(dataset=name, seed=seed, learner=learner, comparator=baseline,
                                      named=bool(named and ratio <= .8),
                                      joint=None if full is None else bool(named and ratio <= .8 and
                                        quality(exact, pd.Series(full['backends'][learner]), task == 'classification'))))
                    checked_pairs += 1
    assert len(table) == vectors
    return dict(cases=len(list(OUT.glob('*/s*/record.json'))), prediction_vectors=vectors,
                metrics=scalars, matched_main_prediction_vectors=copies,
                recomputed_exact_comparisons=checked_pairs, duplicates_in_new_sources=duplicates,
                cnae_filtered_replay=cnae_filtered,
                decisions=joint)


def validate_supplements():
    from experiments.minpi_study_kernel import make_matrix, sampled_rates
    from minppi_mofs.exact_minpi import pivotal_rates

    kernel = pd.read_csv(OUTPUT / 'kernel/curves.csv')
    row = kernel.query("width == 64 and abstain_prob == .1 and seed == 127 and method == 'mc' and draws == 16").iloc[0]
    votes = make_matrix(127, 64, .1)
    exact = pivotal_rates(votes, (0, 4, 16)).min(axis=0)
    sampled = sampled_rates(votes, (0, 4, 16), 16, 127).min(axis=0)
    same(np.abs(exact - sampled).mean(), row.mean_abs_error)
    same(np.abs(exact - sampled).max(), row.max_abs_error)
    cert = pd.read_csv(OUTPUT / 'certificate/replay.csv', keep_default_na=False)
    assert len(cert) == 55 and cert.matches.all()
    har = OUTPUT / 'subject-transfer/seed-17'
    supports = json.loads((har / 'supports.json').read_text(encoding='utf-8'))
    assert supports['exact_10'] == supports['k16_10']
    metrics = pd.read_csv(har / 'metrics.csv')
    with np.load(har / 'predictions.npz') as z:
        for learner in ('ridge', 'knn5'):
            name = f'exact_10_{learner}'
            np.testing.assert_array_equal(z[name], z[f'k16_10_{learner}'])
            y, pred = z['gold'], z[name]
            ba = np.mean([(pred[y == activity] == activity).mean() for activity in np.unique(y)])
            row = metrics.query("method == 'exact' and percent == 10 and learner == @learner and subject == 0").iloc[0]
            same(ba, row.balanced_accuracy)
    return dict(kernel_rows=len(kernel), numerical_kernel_cells_spot_checked=1,
                certificate_support_matches=len(cert), har_prediction_vectors_spot_checked=2)


def main():
    summary = OUTPUT / 'summary'
    csv_options = dict(keep_default_na=False, na_values=[''])
    reported = pd.read_csv(summary / 'split_metrics.csv', **csv_options).set_index(['dataset', 'seed', 'learner', 'method'])
    stability = pd.read_csv(summary / 'support_stability.csv', **csv_options).set_index(['dataset', 'method'])
    pair_table = pd.read_csv(summary / 'paired_comparisons.csv', **csv_options).set_index(['dataset', 'learner', 'method', 'reference'])
    assert reported.index.is_unique and pair_table.index.is_unique
    assert not list((OUTPUT / 'formal').rglob('*failure.json'))
    sources = {}
    observed = []
    duplicate_rows = {}
    checked_supports = 0
    copies = {}
    counts = {}
    record_paths = list((OUTPUT / 'formal').glob('*/s*/record.json'))
    assert len(record_paths) == len(P['datasets']) * len(P['formal_seeds'])
    for dataset in P['datasets']:
        X, y, task, names, order, truth, digest = load_dataset(dataset)
        n, p = X.shape
        assert len(set(names)) == len(names) and set(order) == set(range(p))
        # Compare exact, byte-identical feature rows across train and test.
        hashes = [hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest() for row in X]
        del X
        duplicate_rows[dataset] = {}
        copies[dataset] = dict(k16=0, mean=0, vote=0)
        supports = {arm: [] for arm in P['arms']}
        for seed in P['formal_seeds']:
            folder = OUTPUT / 'formal' / dataset / f's{seed}'
            rec = json.loads((folder / 'record.json').read_text(encoding='utf-8'))
            assert (rec['dataset'], rec['seed'], rec['dataset_source'], rec['task']) == (dataset, seed, digest, task)
            assert (rec['rows'], rec['p'], rec['feature_names']) == (n, p, names)
            np.testing.assert_array_equal(order, rec['original_column_indices'])
            assert set(rec['arms']) == set(P['arms'])
            for name, sha in rec['source_sha256'].items():
                if name not in sources:
                    sources[name] = file_hash(ROOT / name)
                assert sha == sources[name]
            train, test = train_test_split(np.arange(n), test_size=.25, random_state=seed,
                                          stratify=y if task == 'classification' else None)
            train_hashes = {hashes[i] for i in train}
            duplicate_rows[dataset][seed] = sum(hashes[i] in train_hashes for i in test)
            assert (rec['train_rows'], rec['test_rows']) == (len(train), len(test))
            k = ceil(.1 * p)
            with np.load(folder / 'predictions.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['train_indices'], train)
                np.testing.assert_array_equal(z['test_indices'], test)
                np.testing.assert_array_equal(z['y_test'], y[test])
                assert not np.intersect1d(train, test).size and np.union1d(train, test).size == n
                rates = z['exact_rates']
                assert rates.shape == (len(np.unique(np.minimum([0, 4, 16], p - 1))), p)
                assert np.isfinite(rates).all() and np.abs(rates).max() <= 1 + 1e-12
                orders = {'exact': rates.min(axis=0), 'mean': rates.mean(axis=0),
                          'k16': rates[-1], 'vote': rates[0], 'mc32': z['mc32_scores']}
                for arm, score in orders.items():
                    np.testing.assert_array_equal(np.sort(np.argsort(-score, kind='stable')[:k]),
                                                  rec['arms'][arm]['selected'])
                for arm in copies[dataset]:
                    copies[dataset][arm] += rec['arms'][arm]['selected'] == rec['arms']['exact']['selected']
                for arm, data in rec['arms'].items():
                    chosen = data['selected']
                    assert len(chosen) == data['k'] == (p if arm == 'full' else k)
                    assert sorted(set(chosen)) == chosen and 0 <= min(chosen) <= max(chosen) < p
                    original = set(order[chosen].tolist())
                    supports[arm].append(original)
                    checked_supports += 1
                    assert set(data['backends']) == set(P['learners'])
                    for learner, metrics in data['backends'].items():
                        prefix = f'{arm}_{learner}'
                        pred = z[prefix + '_pred']
                        assert pred.shape == (len(test),) and np.isfinite(pred).all()
                        score = z[prefix + '_score'] if task == 'classification' else None
                        if score is not None:
                            assert score.shape[0] == len(test) and np.isfinite(score).all()
                        measured = replay_metrics(y[test], pred, score, learner)
                        row = reported.loc[(dataset, seed, learner, arm)]
                        for key, value in measured.items():
                            same(value, metrics[key])
                            same(value, row[key])
                            counts[key] = counts.get(key, 0) + 1
                        same(len(chosen), row.k)
                        same(len(chosen) / p, row.retained_fraction)
                        if truth:
                            same(len(original.intersection(truth)) / len(truth), row.known_signal_recall)
                        assert data['selector_seconds'] > 0 and metrics['backend_seconds'] > 0
                        elapsed = data['selector_seconds'] + metrics['backend_seconds']
                        same(elapsed, metrics['total_seconds'])
                        same(elapsed, row.total_seconds)
                        observed.append(dict(dataset=dataset, seed=seed, learner=learner, arm=arm,
                                             k=len(chosen), total_seconds=elapsed, **measured))
        for arm, masks in supports.items():
            overlaps = [len(a & b) / len(a | b) for a, b in combinations(masks, 2)]
            comparison = stability.loc[(dataset, arm)]
            same([len(overlaps), np.mean(overlaps), min(overlaps), max(overlaps)],
                 [comparison.pairs, comparison.mean_jaccard, comparison.minimum_jaccard, comparison.maximum_jaccard])
    assert len(observed) == len(reported)
    decisions = paired_rules(pd.DataFrame(observed), pair_table)
    result = dict(cases=len(record_paths), supports=checked_supports, prediction_vectors=len(observed),
                  metric_replays=counts, checked_paired_rows=len(decisions), code_digests=sources,
                  test_rows_identical_to_a_training_feature_vector=duplicate_rows,
                  exact_support_matches_by_five_splits=copies, decisions=decisions)
    result['chain'] = audit_chain()
    result['supplements'] = validate_supplements()
    (summary / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('Cases, supports, prediction vectors, individual metrics and paired rows:',
          len(record_paths), checked_supports, len(observed), sum(counts.values()), len(decisions))


if __name__ == '__main__':
    main()
