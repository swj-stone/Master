"""Replay six matched scores, their held-out predictions and witness resampling."""
import argparse
import hashlib
import itertools
import json
import tempfile
from math import ceil
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT, P, ROOT, file_hash, load_dataset, metrics, train
from minppi_mofs import exact_minpi


ARMS = ('geometry', 'margin', 'vote', 'k16', 'mean', 'exact')
STABILITY_DATA = ('gisette', 'isolet')
REPETITIONS = 3


def reference_weights(X, y, task, seed):
    """Compute the selector's reference-fold Fisher geometry, without querying neighbours."""
    X, y = np.asarray(X), np.asarray(y)
    ref, _ = exact_minpi._folds(y, task, np.random.default_rng(seed), row_limit=256)
    groups = y[ref]
    if task == 'regression':
        cuts = np.unique(np.quantile(groups, [.2, .4, .6, .8]))
        groups = np.searchsorted(cuts, groups)
    raw = np.asarray(X[ref], dtype=np.float32)
    observed = np.isfinite(raw)
    counts = np.maximum(observed.sum(axis=0), 1)
    center = np.where(observed, raw, 0.).sum(axis=0)/counts
    residual = np.where(observed, raw - center, 0.)
    scale = np.sqrt((residual*residual).sum(axis=0)/counts)
    scale = np.where(scale > 1e-7, scale, 1.).astype(np.float32)
    return exact_minpi._reference_geometry(exact_minpi._scaled(raw, center, scale), groups)


def witness_components(X, y, task, seed):
    """Collect the actual reference geometry and time the production front end."""
    geometry = exact_minpi._reference_geometry
    search = exact_minpi._triplets
    measured = {}

    def capture_geometry(*args):
        measured['prepare_s'] = perf_counter() - start
        t = perf_counter()
        result = geometry(*args)
        measured['geometry_s'] = perf_counter() - t
        measured['weights'] = result
        return result

    def capture_search(*args):
        t = perf_counter()
        result = search(*args)
        measured['triplets_s'] = perf_counter() - t
        return result

    with patch.object(exact_minpi, '_reference_geometry', capture_geometry), \
         patch.object(exact_minpi, '_triplets', capture_search):
        start = perf_counter()
        margins = exact_minpi.witness_margins(X, y, task, seed)
    weights = measured.pop('weights')
    return weights, margins, measured


def one_score(method, weights, margins, tiers):
    if method == 'geometry':
        return weights
    if method == 'margin':
        return margins.mean(axis=0)
    votes = np.sign(margins).astype(np.int8)
    if method == 'vote':
        return votes.mean(axis=0)
    if method == 'k16':
        return exact_minpi.pivotal_rates(votes, tiers[-1:])[0]
    rates = exact_minpi.pivotal_rates(votes, tiers)
    return rates.min(axis=0) if method == 'exact' else rates.mean(axis=0)


def select_all(X, y, task, seed):
    h = ceil(X.shape[1] * P['selection_fraction'])
    tiers = np.unique(np.minimum(P['tiers'], X.shape[1] - 1))
    rng = np.random.default_rng([seed, 72493])
    scores, supports, timings = {}, {}, []
    for repeat in range(REPETITIONS):
        weights, margins, stages = witness_components(X, y, task, seed)
        start_geometry = perf_counter()
        direct_weights = reference_weights(X, y, task, seed)
        direct_geometry = perf_counter()-start_geometry
        np.testing.assert_array_equal(direct_weights, weights)
        for arm in rng.permutation(ARMS):
            start = perf_counter()
            score = one_score(arm, weights, margins, tiers)
            chosen = np.sort(np.argsort(-score, kind='stable')[:h])
            scoring = perf_counter()-start
            if repeat:
                np.testing.assert_array_equal(supports[arm], chosen)
            scores[arm], supports[arm] = score, chosen
            search = 0.0 if arm == 'geometry' else stages['triplets_s']
            front = direct_geometry if arm == 'geometry' else sum(stages.values())
            timings.append(dict(repeat=repeat, arm=arm, prepare_s=stages['prepare_s'] if arm != 'geometry' else np.nan,
                                geometry_s=stages['geometry_s'] if arm != 'geometry' else np.nan, triplets_s=search,
                                direct_geometry_s=direct_geometry if arm == 'geometry' else np.nan,
                                score_rank_s=scoring,
                                selector_s=front+scoring))
    rates = exact_minpi.pivotal_rates(np.sign(margins).astype(np.int8), tiers)
    np.testing.assert_array_equal(scores['vote'], rates[0])
    return scores, supports, timings, rates, tiers, margins


def feature_crossings(key, names, rates, tiers, supports):
    largest = rates[-1]
    minimum = rates.min(axis=0)
    penalty = largest - minimum
    admitted = np.setdiff1d(supports['exact'], supports['k16'])
    removed = np.setdiff1d(supports['k16'], supports['exact'])
    changes, pairs = [], []
    for direction, changed in [('in', admitted), ('out', removed)]:
        for j in changed:
            changes.append(dict(**key, direction=direction, column=int(j), feature=names[j],
                                largest=largest[j], minimum=minimum[j], penalty=penalty[j],
                                first_weakest_layer=int(tiers[rates[:,j].argmin()])))
    for j, l in itertools.product(removed, admitted):
        gap = largest[j]-largest[l]
        deduction = penalty[j]-penalty[l]
        if gap < -1e-14 or deduction < gap-1e-14:
            raise AssertionError('Selected pair contradicts the layer-score order')
        pairs.append(dict(**key, outgoing=names[j], incoming=names[l], largest_gap=gap,
                          penalty_difference=deduction, minimum_reversal=minimum[l]-minimum[j]))
    return changes, pairs


def load_split(dataset, seed, grouped):
    if grouped:
        group_file = OUTPUT/'summary/spambase_grouped_predictions.npz'
        with np.load(group_file) as data:
            prefix = f's{seed}_'
            return {key[len(prefix):]: data[key] for key in data.files if key.startswith(prefix)}, None
    case = OUTPUT/'formal'/dataset/f's{seed}'
    record = json.loads((case/'record.json').read_text(encoding='utf-8'))
    with np.load(case/'predictions.npz') as data:
        keep = ('train_indices', 'test_indices', 'y_test', 'exact_rates')
        saved = {key: data[key] for key in data.files
                 if key in keep or any(key.startswith(arm+'_') for arm in ARMS[1:])}
    return saved, record


def verify_prediction(archive, arm, learner, pred, score):
    prefix = f'{arm}_{learner}'
    if prefix+'_pred' not in archive:
        return False
    np.testing.assert_allclose(pred, archive[prefix+'_pred'], rtol=0, atol=1e-10)
    if score is not None:
        np.testing.assert_allclose(score, archive[prefix+'_score'], rtol=0, atol=1e-10)
    return True


def evaluate(panel, dataset, seed, data, folder):
    X, y, task, names, original, _, source = data
    grouped = panel == 'grouped'
    archive, record = load_split(dataset, seed, grouped)
    a, b = archive['train_indices'], archive['test_indices']
    np.testing.assert_array_equal(y[b], archive['truth' if grouped else 'y_test'])
    assert not np.intersect1d(a, b).size
    if record is not None and (source != record['dataset_source'] or names != record['feature_names']):
        raise ValueError('Original feature or source identity changed')
    Xtr, Xte, ytr = X[a], X[b], y[a]
    scores, supports, timing, rates, tiers, margins = select_all(Xtr, ytr, task, seed)
    key = dict(panel=panel, dataset=dataset, seed=seed)
    values = dict(train_indices=a, test_indices=b, y_test=y[b], tiers=tiers,
                  rates=rates, feature_names=np.asarray(names), original_columns=original)
    if record is not None:
        np.testing.assert_allclose(rates, archive['exact_rates'], rtol=0, atol=1e-13)
    else:
        np.testing.assert_array_equal(original[supports['exact']], archive['exact_original_features'])
    rows = []
    rng = np.random.default_rng([seed, 54319])
    for arm in rng.permutation(ARMS):
        cols = supports[arm]
        if record is not None and arm != 'geometry':
            np.testing.assert_array_equal(cols, record['arms'][arm]['selected'])
        values[arm+'_scores'] = scores[arm]
        values[arm+'_support'] = cols
        for learner in rng.permutation(P['learners']):
            start = perf_counter()
            if grouped:
                train_x, test_x = X[np.ix_(a, cols)], X[np.ix_(b, cols)]
            else:
                train_x, test_x = Xtr[:,cols], Xte[:,cols]
            pred, score = train(learner, train_x, ytr, test_x, task, seed)
            fit = perf_counter()-start
            verified = verify_prediction(archive, arm, learner, pred, score)
            values[f'{arm}_{learner}_pred'] = pred
            if score is not None:
                values[f'{arm}_{learner}_score'] = score
            time_select = np.median([r['selector_s'] for r in timing if r['arm'] == arm])
            h = len(cols)
            intersect = len(np.intersect1d(cols, supports['exact']))
            rows.append(dict(**key, arm=arm, learner=learner, h=h, p=X.shape[1],
                             n_test=len(b), witnesses=len(margins),
                             jaccard_to_min=intersect/(2*h-intersect),
                             selector_s=time_select, backend_s=fit, total_s=time_select+fit,
                             archived_prediction_checked=verified, **metrics(y[b],pred,score)))
    changes, pairs = feature_crossings(key, names, rates, tiers, supports)
    target = folder/f'{panel}-{dataset}-s{seed}.npz'
    np.savez_compressed(target, **values)
    manifest = dict(**key, data_source=source, file=target.name, sha256=file_hash(target),
                    witness_sha256=hashlib.sha256(margins.tobytes()).hexdigest(),
                    h=len(supports['exact']), switched=len(changes)//2, tiers=tiers.tolist())
    return rows, [dict(**key, **t) for t in timing], changes, pairs, manifest


def resample(dataset, data, folder):
    """Use all 25 training-split by witness-seed cells on each wide source."""
    X, y, task, _, _, _, _ = data
    seeds = P['formal_seeds']
    h = ceil(P['selection_fraction']*X.shape[1])
    tiers = np.unique(np.minimum(P['tiers'], X.shape[1]-1))
    supp = {}
    ar = {}
    for train_seed in seeds:
        archive, record = load_split(dataset, train_seed, False)
        a = archive['train_indices']
        for witness_seed in seeds:
            delta = exact_minpi.witness_margins(X[a], y[a], task, witness_seed)
            rate = exact_minpi.pivotal_rates(np.sign(delta).astype(np.int8), tiers)
            cols = np.sort(np.argsort(-rate.min(axis=0), kind='stable')[:h])
            if train_seed == witness_seed:
                np.testing.assert_array_equal(cols, record['arms']['exact']['selected'])
            supp[train_seed,witness_seed] = cols
            ar[f't{train_seed}_w{witness_seed}'] = cols
    np.savez_compressed(folder/f'stability-{dataset}.npz', **ar)
    result = []
    for (ta,wa), (tb,wb) in itertools.combinations(supp, 2):
        if ta == tb:
            family = 'within_train'
        elif wa == wb:
            family = 'across_train_same_rng'
        elif ta == wa and tb == wb:
            family = 'archived_diagonal'
        else:
            continue
        intersect = len(np.intersect1d(supp[ta,wa], supp[tb,wb]))
        result.append(dict(dataset=dataset, contrast=family, train_a=ta, witness_a=wa,
                           train_b=tb, witness_b=wb, jaccard=intersect/(2*h-intersect)))
    return result


def publish_comparisons(folder):
    quality = pd.read_csv(folder/'quality.csv')
    swaps = pd.read_csv(folder/'swaps.csv')
    contrasts, regression = [], []
    for (panel, dataset, seed, learner), variants in quality.groupby(['panel', 'dataset', 'seed', 'learner']):
        key = dict(panel=panel, dataset=dataset, seed=seed, learner=learner)
        arms = variants.set_index('arm')
        metric = 'r2' if np.isfinite(arms.iloc[0]['r2']) else 'auc'
        for left, right in [('margin','geometry'), ('vote','margin'), ('k16','vote'),
                            ('exact','k16'), ('exact','mean'), ('exact','geometry')]:
            a, b = arms.loc[left], arms.loc[right]
            contrasts.append(dict(**key, first=left, second=right, metric=metric,
                                  first_value=a[metric], second_value=b[metric],
                                  quality_delta=a[metric]-b[metric],
                                  mae_delta=a['mae']-b['mae'], seconds_delta=a['total_s']-b['total_s']))
        if metric == 'r2':
            changed = swaps.loc[(swaps.dataset==dataset)&(swaps.seed==seed)&(swaps.panel==panel)]
            smallest, largest = arms.loc['exact'], arms.loc['k16']
            regression.append(dict(**key,
                                   admitted=', '.join(changed[changed.direction=='in'].feature),
                                   removed=', '.join(changed[changed.direction=='out'].feature),
                                   largest_r2=largest['r2'], minimum_r2=smallest['r2'],
                                   delta_r2=smallest['r2']-largest['r2'],
                                   largest_mae=largest['mae'], minimum_mae=smallest['mae'],
                                   delta_mae=smallest['mae']-largest['mae']))
    pd.DataFrame(contrasts).to_csv(folder/'paired.csv', index=False)
    pd.DataFrame(regression).to_csv(folder/'regression_changes.csv', index=False)


def run(out):
    if out.exists():
        raise FileExistsError(f'Choose a new output path: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, times, swaps, pairs, cases, stability = [], [], [], [], [], []
    with tempfile.TemporaryDirectory(dir=out.parent) as temp, threadpool_limits(limits=1):
        folder = Path(temp)/'matched'
        folder.mkdir()
        for name in P['primary_sources']:
            data = load_dataset(name)
            panels = ('formal', 'grouped') if name == 'spambase' else ('formal',)
            for panel, seed in itertools.product(panels, P['formal_seeds']):
                r, ts, sw, cs, manifest = evaluate(panel, name, seed, data, folder)
                rows.extend(r)
                times.extend(ts)
                swaps.extend(sw)
                pairs.extend(cs)
                cases.append(manifest)
            if name in STABILITY_DATA:
                stability.extend(resample(name, data, folder))
            print(name, 'done', flush=True)
        for basename, records in [('quality',rows), ('stage_timings',times), ('swaps',swaps),
                                  ('swap_contrasts',pairs), ('stability',stability)]:
            pd.DataFrame(records).to_csv(folder/f'{basename}.csv', index=False)
        publish_comparisons(folder)
        sources = [Path(__file__), ROOT/'experiments/minpi_study.py', ROOT/'configs/minpi-study.json',
                   ROOT/'src/minppi_mofs/exact_minpi.py', ROOT/'src/minppi_mofs/piv3_compact_filter.py',
                   ROOT/'src/minppi_mofs/piv2.py']
        metadata = dict(cases=cases, threads=1, score_timing_repeats=REPETITIONS,
                        source_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources})
        (folder/'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        folder.rename(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
