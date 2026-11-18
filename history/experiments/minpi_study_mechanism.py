"""Evaluate pivotal voting on independently generated, noisy matched comparisons."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from minppi_mofs.exact_minpi import pivotal_rates


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/minpi-study/mechanism-protocol.md'
TIERS = (0, 4, 16)
REGIMES = ('valid_divided', 'reversed_divided')
METHODS = ('sign', 'largest', 'minimum')
H = 6
WORLDS = 48


def signs(rng, shape):
    return 2 * rng.integers(0, 2, size=shape, dtype=np.int8) - 1


def sample_measurements(rng, factors, types, loadings):
    return factors[:, types] * loadings + rng.standard_normal((len(factors), len(types)))


def triplets(rng, size, regime, types, loadings):
    query = signs(rng, (size, 3))
    valid = rng.random(size) < .5
    structured = ~valid if regime == 'valid_divided' else valid
    hit = query.copy()
    hit[:, 0] *= np.where(valid, 1, -1).astype(np.int8)
    hit[:, 1] *= np.where(valid, -1, 1).astype(np.int8)
    hit[structured, 2] *= signs(rng, structured.sum())
    miss = -hit
    miss[~structured, 2] = query[~structured, 2]
    q, h, m = (sample_measurements(rng, part, types, loadings) for part in (query, hit, miss))
    votes = np.sign(np.abs(q - m) - np.abs(q - h)).astype(np.int8)
    imbalance = np.abs(votes.mean(axis=1))
    return votes, {
        'valid_rate': valid.mean(),
        'structured_imbalance': imbalance[structured].mean(),
        'unstructured_imbalance': imbalance[~structured].mean(),
    }


def prediction_sample(rng, size, types, loadings):
    factors = signs(rng, (size, 3))
    return sample_measurements(rng, factors, types, loadings), (factors[:, 0] > 0).astype(int)


def world(regime_ix, world_ix):
    streams = np.random.SeedSequence([20261117, regime_ix, world_ix]).spawn(5)
    layout, train, assess, fit, test = map(np.random.default_rng, streams)
    types = layout.permutation(np.repeat(np.arange(3), [6, 6, 48]))
    loadings = np.where(types == 2, layout.uniform(1.5, 2.5, size=60),
                        layout.uniform(.65, 1.25, size=60))
    regime = REGIMES[regime_ix]
    votes, train_stats = triplets(train, 192, regime, types, loadings)
    evaluation, eval_stats = triplets(assess, 768, regime, types, loadings)
    tiers = pivotal_rates(votes, TIERS)
    check_tiers = pivotal_rates(evaluation, TIERS)
    scores = (votes.mean(axis=0), tiers[-1], tiers.min(axis=0))
    if not np.array_equal(scores[0], tiers[0]):
        raise ValueError('Size zero should equal the mean raw ballot')
    supports = [np.sort(np.argsort(-s, kind='stable')[:H]) for s in scores]
    X, y = prediction_sample(fit, 384, types, loadings)
    X_test, y_test = prediction_sample(test, 1536, types, loadings)

    results, predictions = [], []
    for method, support in zip(METHODS, supports, strict=True):
        learner = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver='lbfgs'))
        learner.fit(X[:, support], y)
        probability = learner.predict_proba(X_test[:, support])[:, 1]
        predictions.append(probability)
        results.append({
            'regime': regime, 'world': world_ix, 'method': method,
            'support': ' '.join(map(str, support)), 'signal_count': int((types[support] == 0).sum()),
            'site_count': int((types[support] == 1).sum()),
            'worst_layer': check_tiers.min(axis=0)[support].mean(),
            'auc': roc_auc_score(y_test, probability), 'logloss': log_loss(y_test, probability),
            'swaps_from_largest': H - len(np.intersect1d(support, supports[1])),
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'assess_{k}': v for k, v in eval_stats.items()},
        })
    return results, (types, loadings, votes, evaluation, tiers, check_tiers, y_test, predictions)


def contrasts(data):
    rng = np.random.default_rng(20261118)
    rows = []
    for regime in REGIMES:
        subset = data.loc[data.regime == regime]
        indices = rng.integers(WORLDS, size=(10000, WORLDS))
        for lhs, rhs in [('largest', 'sign'), ('minimum', 'largest')]:
            for metric in ['auc', 'logloss', 'signal_count', 'worst_layer']:
                paired = subset.pivot(index='world', columns='method', values=metric)
                delta = (paired[lhs] - paired[rhs]).to_numpy()
                primary = lhs == 'largest' and metric == 'auc'
                alpha = .025 if primary else .05
                low, high = np.quantile(delta[indices].mean(axis=1), [alpha / 2, 1 - alpha / 2])
                rows.append(dict(regime=regime, contrast=f'{lhs}-{rhs}', metric=metric,
                                 worlds=len(delta), mean=delta.mean(), low=low, high=high,
                                 coverage=1-alpha, wins=int((delta > 0).sum())))
    return pd.DataFrame(rows)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    keys = ('types', 'loadings', 'train_votes', 'assess_votes', 'train_tiers', 'assess_tiers',
            'test_y', 'probabilities')
    arrays = {k: [] for k in keys}
    records = []
    with threadpool_limits(limits=1):
        for i in range(len(REGIMES)):
            for w in range(WORLDS):
                results, values = world(i, w)
                records.extend(results)
                for k, value in zip(keys, values, strict=True):
                    arrays[k].append(value)
    data = pd.DataFrame(records)
    data.to_csv(output / 'worlds.csv', index=False)
    contrasts(data).to_csv(output / 'contrasts.csv', index=False)
    np.savez_compressed(output / 'observations.npz', **arrays, regime_order=REGIMES,
                        worlds_per_regime=WORLDS, method_order=METHODS, tiers=TIERS)
    metadata = (f'protocol_sha256 {sha256(PROTOCOL.read_bytes()).hexdigest()}\n'
                f'driver_sha256 {sha256(Path(__file__).read_bytes()).hexdigest()}\n'
                'SeedSequence entropy = [20261117, regime_index, world_index]\n'
                'Archive first axis = regime_index * 48 + world_index. Features are zero-indexed.\n'
                'Types 0/1/2 correspond to class/site/background.\n')
    (output / 'readme.txt').write_text(metadata, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/minpi-study/mechanism')
    main(parser.parse_args().output)
