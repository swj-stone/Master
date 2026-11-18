"""Frozen wrist/ankle transfer on ten MHEALTH participants (UCI DOI 10.24432/C5TW22)."""

import argparse
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from minppi_mofs.exact_minpi import pivotal_rates


ROOT = Path(__file__).resolve().parents[1]
RAW_SHA256 = '16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0'
SOURCE = 'https://archive.ics.uci.edu/static/public/319/mhealth%2Bdataset.zip'
DATA = ROOT / 'data/revision-cohorts'
TIERS = (0, 4, 16)
ACTIVITIES = np.arange(1, 13)
STATS = ('mean', 'sd', 'min', 'max', 'q25', 'median', 'q75', 'rms', 'absdiff', 'meanabs')
FEATURES = [f'{axis}_{s}' for axis in ('x', 'y', 'z', 'norm') for s in STATS]


def features(window):
    v = np.column_stack((window, np.linalg.norm(window, axis=1)))
    q25, q50, q75 = np.quantile(v, [.25, .5, .75], axis=0)
    stats = (v.mean(axis=0), v.std(axis=0), v.min(axis=0), v.max(axis=0), q25, q50, q75,
             np.sqrt((v*v).mean(axis=0)), np.abs(np.diff(v, axis=0)).mean(axis=0),
             np.abs(v).mean(axis=0))
    return np.column_stack(stats).ravel()


def extract(archive_path, data_path):
    if sha256(archive_path.read_bytes()).hexdigest() != RAW_SHA256:
        raise ValueError('MHEALTH source checksum mismatch')
    rows, values = [], []
    with ZipFile(archive_path) as archive:
        for subject in range(1, 11):
            with archive.open(f'MHEALTHDATASET/mHealth_subject{subject}.log') as source:
                raw = np.loadtxt(source, usecols=(14, 15, 16, 5, 6, 7, 23))
            y = raw[:, -1].astype(int)
            boundaries = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
                if y[start] not in ACTIVITIES:
                    continue
                for offset in range(start, end - 99, 100):
                    rows.append((subject, y[offset], offset))
                    window = raw[offset:offset+100]
                    values.append([features(window[:, :3]), features(window[:, 3:6])])
    with data_path.open('xb') as dest:
        np.savez_compressed(dest, windows=values, ids=rows, feature_names=FEATURES,
                            devices=['right wrist', 'left ankle'], raw_sha256=RAW_SHA256)


def banks(ids, members, rng):
    reference, query = [], []
    for person in np.unique(ids[members, 0]):
        for label in ACTIVITIES:
            eligible = members[(ids[members, 0] == person) & (ids[members, 1] == label)]
            shuffled = rng.permutation(eligible)
            middle = len(shuffled) // 2
            reference.extend(shuffled[:middle])
            query.extend(shuffled[middle:])
    return np.sort(reference).astype(int), np.sort(query).astype(int)


def ballot(X, ids, reference, query, cap, rng, exclude_person):
    queries = np.concatenate([rng.permutation(query[ids[query, 1] == label])[:cap]
                              for label in ACTIVITIES])
    ankle = rng.permutation(np.arange(len(queries)) % 3 != 0)
    triplets, votes = [], []
    for q, cross in zip(queries, ankle, strict=True):
        device = int(cross)
        ref = reference[ids[reference, 0] != ids[q, 0]] if exclude_person else reference
        delta = X[ref, device] - X[q, 0]
        distances = (delta * delta).sum(axis=1)
        same = ids[ref, 1] == ids[q, 1]
        if not same.any() or same.all() or q in ref:
            raise ValueError('Missing reference class or overlapping raw windows')
        h = ref[np.argmin(np.where(same, distances, np.inf))]
        m = ref[np.argmin(np.where(~same, distances, np.inf))]
        votes.append(np.sign(np.abs(X[m, device]-X[q, 0]) - np.abs(X[h, device]-X[q, 0])))
        triplets.append((q, h, m, device))
    return np.asarray(votes, dtype=np.int8), np.asarray(triplets)


def evaluate(data_path, output):
    output.mkdir(parents=True, exist_ok=False)
    with np.load(data_path, allow_pickle=False) as source:
        X, ids = source['windows'], source['ids']
        if (source['raw_sha256'].item() != RAW_SHA256 or list(source['feature_names']) != FEATURES
                or np.unique(ids, axis=0).shape != ids.shape or not np.isfinite(X).all()):
            raise ValueError('Inconsistent derived-window archive')
    rngs = list(map(np.random.default_rng, np.random.SeedSequence(20261123).spawn(15)))
    dev, test = np.flatnonzero(ids[:, 0] <= 5), np.flatnonzero(ids[:, 0] >= 6)
    ref, query = banks(ids, dev, rngs[0])
    scaler = StandardScaler().fit(X[ref].reshape(-1, 40))
    scaled = scaler.transform(X.reshape(-1, 40)).reshape(X.shape)
    votes, comparisons = ballot(scaled, ids, ref, query, 16, rngs[1], True)
    rates = pivotal_rates(votes, TIERS)
    devices = (rngs[2].random(len(dev)) < 2/3).astype(int)
    train = X[dev, devices]
    labels = ids[dev, 1]
    f_scores = f_classif(train, labels)[0]
    raw = dict(sign=rates[0], largest=rates[-1], minimum=rates.min(axis=0), F=f_scores)
    supports = {method: np.sort(np.argsort(-scores, kind='stable')[:4]) for method,scores in raw.items()}
    supports['random'] = np.sort(rngs[3].choice(40, size=4, replace=False))
    supports['full'] = np.arange(40)
    arrays = dict(dev=dev, test=test, train_ref=ref, train_queries=query, train_votes=votes,
                  train_comparisons=comparisons, train_device=devices, tiers=TIERS, rates=rates)
    heldout = {}
    for i, person in enumerate(range(6, 11)):
        ix = np.flatnonzero(ids[:, 0] == person)
        ref, query = banks(ids, ix, rngs[4+2*i])
        votes, triples = ballot(scaled, ids, ref, query, 4, rngs[5+2*i], False)
        heldout[person] = pivotal_rates(votes, TIERS).min(axis=0)
        arrays.update({f'p{person}_ref':ref, f'p{person}_queries':query, f'p{person}_votes':votes,
                       f'p{person}_comparisons':triples, f'p{person}_weakest':heldout[person]})
    results, support_rows = [], []
    with threadpool_limits(limits=1):
        for method, support in supports.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver='lbfgs', max_iter=2000))
            model.fit(train[:, support], labels)
            predictions = np.stack([model.predict(X[test, device][:, support]) for device in (0, 1)])
            arrays[f'{method}_predictions'] = predictions
            support_rows.append(dict(method=method, indices=' '.join(map(str, support)),
                                     names=' '.join(FEATURES[i] for i in support)))
            for person, scores in heldout.items():
                member = ids[test, 0] == person
                truth = ids[test, 1][member]
                ba = [balanced_accuracy_score(truth, pred[member]) for pred in predictions]
                results.append(dict(method=method, subject=person, windows=member.sum(),
                                    labels=len(np.unique(truth)), wrist_ba=ba[0], ankle_ba=ba[1],
                                    lower_ba=min(ba), assessment_floor=scores[support].mean()))
    pd.DataFrame(results).to_csv(output/'by-subject.csv', index=False)
    pd.DataFrame(support_rows).to_csv(output/'supports.csv', index=False)
    np.savez_compressed(output/'records.npz', **arrays)
    protocol = ROOT / 'docs/minpi-study/device-protocol.md'
    metadata = (f'MHEALTH (Banos, Garcia and Saez; UCI 2014; DOI 10.24432/C5TW22; CC BY 4.0).\n'
                f'Source {SOURCE}\nRaw SHA256 {RAW_SHA256}\n'
                f'Derived SHA256 {sha256(data_path.read_bytes()).hexdigest()}\n'
                f'Protocol SHA256 {sha256(protocol.read_bytes()).hexdigest()}\n'
                f'Driver SHA256 {sha256(Path(__file__).read_bytes()).hexdigest()}\n'
                f'Development / test windows {len(dev)} / {len(test)}\n'
                f'Training comparisons {len(comparisons)}; cross-instrument {comparisons[:,3].sum()}\n'
                'Rows are nonoverlapping homogeneous activity windows with 0-based recording offsets.\n'
                'Comparison rows are derived-window indices [query,hit,miss,reference_device].\n')
    (output/'source.txt').write_text(metadata, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', action='store_true', help='extract derived features from the official zip')
    parser.add_argument('--raw', type=Path, default=DATA/'mhealth.zip')
    parser.add_argument('--windows', type=Path, default=DATA/'mhealth-windows.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'results/minpi-study/device')
    args = parser.parse_args()
    if args.extract:
        extract(args.raw, args.windows)
    else:
        evaluate(args.windows, args.output)
