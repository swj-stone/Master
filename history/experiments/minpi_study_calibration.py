"""Prospective two-instrument follow-up to the matched-comparison simulation."""

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

from experiments.minpi_study_mechanism import H, METHODS, ROOT, TIERS, WORLDS, signs
from minppi_mofs.exact_minpi import pivotal_rates


def measurements(rng, y, background, cross_instrument, types, loadings):
    polarity = np.where(cross_instrument, -y, y)
    factors = np.column_stack([y, polarity, background])
    return factors[:, types] * loadings + rng.standard_normal((len(y), len(types)))


def triplets(rng, n, types, loadings, background_match=None):
    y, background = signs(rng, (2, n))
    cross = rng.random(n) < 2/3
    background_h = background.copy()
    background_m = background.copy()
    if background_match is None:
        alignment = signs(rng, cross.sum())
    else:
        alignment = np.where(rng.random(cross.sum()) < background_match, 1, -1).astype(np.int8)
    background_h[cross] *= alignment
    background_m[cross] = -background_h[cross]
    query = measurements(rng, y, background, False, types, loadings)
    hit = measurements(rng, y, background_h, cross, types, loadings)
    miss = measurements(rng, -y, background_m, cross, types, loadings)
    return np.sign(np.abs(miss - query) - np.abs(hit - query)).astype(np.int8), cross


def individuals(rng, n, types, loadings):
    y, background = signs(rng, (2, n))
    instrument = rng.random(n) < 2/3
    X = measurements(rng, y, background, instrument, types, loadings)
    return X, (y > 0).astype(int), instrument


def world(index, *, entropy=20261119, background_match=None):
    seeds = np.random.SeedSequence([entropy, index]).spawn(5)
    layout, train, assess, fit, test = map(np.random.default_rng, seeds)
    types = layout.permutation(np.repeat(np.arange(3), [6, 6, 48]))
    loadings = layout.uniform(np.choose(types, [.55, 1.30, 1.50]),
                              np.choose(types, [.85, 1.70, 2.50]))
    votes, cross = triplets(train, 192, types, loadings, background_match)
    evaluation, cross_assess = triplets(assess, 768, types, loadings, background_match)
    rates = pivotal_rates(votes, TIERS)
    assessment = pivotal_rates(evaluation, TIERS)
    scores = (rates[0], rates[-1], rates.min(axis=0))
    supports = [np.sort(np.argsort(-s, kind='stable')[:H]) for s in scores]
    X, y, _ = individuals(fit, 384, types, loadings)
    X_test, y_test, b = individuals(test, 1536, types, loadings)
    results, probabilities = [], []
    for method, selected in zip(METHODS, supports, strict=True):
        learner = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver='lbfgs'))
        learner.fit(X[:, selected], y)
        pred = learner.predict_proba(X_test[:, selected])[:, 1]
        probabilities.append(pred)
        a_auc, b_auc = (roc_auc_score(y_test[idx], pred[idx]) for idx in (~b, b))
        results.append(dict(world=index, method=method, support=' '.join(map(str, selected)),
                            swaps_from_largest=H-len(np.intersect1d(selected, supports[1])),
                            stable_count=int((types[selected] == 0).sum()),
                            reversing_count=int((types[selected] == 1).sum()),
                            worst_layer=assessment.min(axis=0)[selected].mean(),
                            auc=roc_auc_score(y_test, pred), logloss=log_loss(y_test, pred),
                            auc_A=a_auc, auc_B=b_auc, worst_instrument_auc=min(a_auc, b_auc)))
    return results, dict(types=types, loadings=loadings, train_votes=votes, assess_votes=evaluation,
                         train_instrument=cross, assess_instrument=cross_assess,
                         train_tiers=rates, assess_tiers=assessment, test_y=y_test,
                         test_instrument=b, probabilities=probabilities)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    records, observations = [], []
    with threadpool_limits(limits=1):
        for i in range(WORLDS):
            rows, data = world(i)
            records.extend(rows)
            observations.append(data)
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'worlds.csv', index=False)
    paired = []
    indices = np.random.default_rng(20261120).integers(WORLDS, size=(10000, WORLDS))
    for rhs in ('largest', 'sign'):
        for metric in ('worst_layer', 'worst_instrument_auc', 'stable_count', 'auc', 'logloss'):
            table = frame.pivot(index='world', columns='method', values=metric)
            delta = (table['minimum'] - table[rhs]).to_numpy()
            primary = rhs == 'largest' and metric in ('worst_layer', 'worst_instrument_auc')
            alpha = .025 if primary else .05
            low, high = np.quantile(delta[indices].mean(axis=1), [alpha / 2, 1-alpha / 2])
            paired.append(dict(contrast=f'minimum-{rhs}', metric=metric, mean=delta.mean(),
                               low=low, high=high, worlds=WORLDS, coverage=1-alpha))
    pd.DataFrame(paired).to_csv(output / 'contrasts.csv', index=False)
    np.savez_compressed(output / 'observations.npz', method_order=METHODS, tiers=TIERS,
                        **{key: [x[key] for x in observations] for key in observations[0]})
    protocol = ROOT / 'docs/minpi-study/calibration-protocol.md'
    metadata = (f'protocol_sha256 {sha256(protocol.read_bytes()).hexdigest()}\n'
                f'driver_sha256 {sha256(Path(__file__).read_bytes()).hexdigest()}\n'
                'Worlds in ascending index. Instrument B is True.\n'
                'Feature types 0/1/2 are stable/reversing/background.\n')
    (output / 'readme.txt').write_text(metadata, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/minpi-study/mechanism/calibration')
    main(parser.parse_args().output)
