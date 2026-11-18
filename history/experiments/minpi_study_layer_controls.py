"""Post hoc layer, mean, F and random controls on the frozen simulation samples."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.minpi_study_calibration import individuals
from experiments.minpi_study_device import DATA, FEATURES
from experiments.minpi_study_mechanism import H, ROOT, WORLDS, prediction_sample


def simulated(folder):
    jobs = [('valid_divided', folder/'observations.npz', 0),
            ('reversed_divided', folder/'observations.npz', 1),
            ('calibration', folder/'calibration/observations.npz', None)]
    jobs.extend((f'policy-{i}', folder/f'acquisition-bias/policy-{i}.npz', None) for i in range(4))
    rows = []
    for name, path, regime in jobs:
        with np.load(path, allow_pickle=False) as archive:
            types, loadings = archive['types'], archive['loadings']
            rates, assessment = archive['train_tiers'], archive['assess_tiers'].min(axis=1)
            y_check, pred_check = archive['test_y'], archive['probabilities']
            instrument_check = archive['test_instrument'] if regime is None else None
        for w in range(WORLDS):
            if regime is None:
                ix = w
                entropy = 20261119 if name == 'calibration' else 20261121
                seeds = np.random.SeedSequence([entropy, w]).spawn(5)
                X, y, _ = individuals(np.random.default_rng(seeds[3]), 384, types[ix], loadings[ix])
                Xt, yt, b = individuals(np.random.default_rng(seeds[4]), 1536, types[ix], loadings[ix])
                np.testing.assert_array_equal(instrument_check[ix], b)
            else:
                ix = regime*WORLDS+w
                seeds = np.random.SeedSequence([20261117, regime, w]).spawn(5)
                X, y = prediction_sample(np.random.default_rng(seeds[3]), 384, types[ix], loadings[ix])
                Xt, yt = prediction_sample(np.random.default_rng(seeds[4]), 1536, types[ix], loadings[ix])
                b = None
            np.testing.assert_array_equal(yt, y_check[ix])
            scores = dict(G4=rates[ix, 1], average=rates[ix].mean(axis=0), min=rates[ix].min(axis=0),
                          fit_F=f_classif(X, y)[0], random=np.random.default_rng(90271+w).random(60))
            for method, s in scores.items():
                selected = np.sort(np.argsort(-s, kind='stable')[:H])
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver='lbfgs'))
                pred = model.fit(X[:,selected], y).predict_proba(Xt[:,selected])[:,1]
                if method == 'min':
                    np.testing.assert_allclose(pred, pred_check[ix,2], rtol=0, atol=1e-12)
                worse_auc = (min(roc_auc_score(yt[g], pred[g]) for g in (b,~b)) if b is not None
                             else np.nan)
                rows.append(dict(condition=name, world=w, method=method, auc=roc_auc_score(yt,pred),
                                 worst_auc=worse_auc, held_min=assessment[ix,selected].mean(),
                                 logloss=log_loss(yt,pred), support=' '.join(map(str,selected))))
    return pd.DataFrame(rows)


def device(folder):
    with np.load(DATA/'mhealth-windows.npz') as data:
        X, ids = data['windows'], data['ids']
    rows = []
    with np.load(folder/'records.npz') as archive:
        dev, test, site = archive['dev'], archive['test'], archive['train_device']
        rates = archive['rates']
        scores = dict(G4=rates[1], average=rates.mean(axis=0), minimum=rates.min(axis=0))
        for method, s in scores.items():
            support = np.sort(np.argsort(-s, kind='stable')[:4])
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver='lbfgs', max_iter=2000))
            model.fit(X[dev,site][:,support], ids[dev,1])
            pred = np.stack([model.predict(X[test,d][:,support]) for d in (0,1)])
            if method == 'minimum':
                np.testing.assert_array_equal(pred, archive['minimum_predictions'])
            for person in range(6, 11):
                mask = ids[test,0] == person
                ba = [balanced_accuracy_score(ids[test,1][mask], labels[mask]) for labels in pred]
                rows.append(dict(method=method, subject=person, wrist_ba=ba[0], ankle_ba=ba[1],
                                 lower_ba=min(ba), assessment_floor=archive[f'p{person}_weakest'][support].mean(),
                                 support=' '.join(FEATURES[i] for i in support)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=('simulated','device'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        folder = ROOT/'results/minpi-study'
        data = simulated(folder/'mechanism') if args.study == 'simulated' else device(folder/'device')
        data.to_csv(args.output, mode='x', index=False)
