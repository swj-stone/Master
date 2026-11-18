"""Recompute simulated endpoints and wrist/ankle leakage checks from release arrays."""

from functools import lru_cache
from hashlib import sha256
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

from experiments.minpi_study_device import DATA, FEATURES
from experiments.minpi_study_mechanism import METHODS, ROOT, TIERS, WORLDS


@lru_cache(None)
def exact_row_weight(pos, neg, zero, k, sign):
    """Count feasible permutations by integer binomial coefficients."""
    pos -= int(sign > 0)
    neg -= int(sign < 0)
    numerator = 0
    for a in range(min(pos, k)+1):
        for b in range(min(neg, k-a)+1):
            c = k-a-b
            if c <= zero and a-b in (0,-sign):
                numerator += comb(pos,a)*comb(neg,b)*comb(zero,c)
    return numerator / comb(pos+neg+zero,k)


def check_rates(matrix, expected):
    counts = np.column_stack([(matrix == s).sum(axis=1) for s in (1,-1,0)])
    actual = np.zeros(expected.shape)
    for i,k in enumerate(TIERS):
        for sign in (1,-1):
            weights = [exact_row_weight(int(p),int(n),int(z),k,sign)
                       if (p if sign == 1 else n) else 0. for p,n,z in counts]
            actual[i] += sign*(np.asarray(weights) @ (matrix == sign))/len(matrix)
    np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-15)


def simulated(folder, npz_name='observations.npz', subset=None):
    data = pd.read_csv(folder/'worlds.csv')
    if subset is not None:
        data = data[np.isclose(data.background_match,subset)].copy()
    data = data.reset_index(drop=True)
    with np.load(folder/npz_name, allow_pickle=False) as raw:
        tiers = raw['train_tiers']
        assessed = raw['assess_tiers'].min(axis=1)
        yt, probabilities = raw['test_y'], raw['probabilities']
        b = raw['test_instrument'] if 'test_instrument' in raw else None
        if len(tiers) not in (WORLDS,2*WORLDS) or len(data) != len(tiers)*len(METHODS):
            raise ValueError('Unexpected world or output count')
        check_rates(raw['train_votes'][0], tiers[0])
        check_rates(raw['assess_votes'][-1], raw['assess_tiers'][-1])
    for i,row in data.iterrows():
        w,method = divmod(i,len(METHODS))
        assert row['method'] == METHODS[method] and row.world == w % WORLDS
        scores = (tiers[w,0],tiers[w,-1],tiers[w].min(axis=0))[method]
        selected = np.sort(np.argsort(-scores,kind='stable')[:6])
        np.testing.assert_array_equal(selected,np.fromstring(row.support,dtype=int,sep=' '))
        pred = probabilities[w,method]
        np.testing.assert_allclose([assessed[w,selected].mean(),roc_auc_score(yt[w],pred),log_loss(yt[w],pred)],
                                   [row.worst_layer,row.auc,row.logloss],rtol=0,atol=1e-12)
        if b is not None:
            values = [roc_auc_score(yt[w,g],pred[g]) for g in (~b[w],b[w])]
            np.testing.assert_allclose([*values,min(values)],[row.auc_A,row.auc_B,row.worst_instrument_auc],
                                       rtol=0,atol=1e-12)
    return len(data)


def device(folder):
    derived = DATA/'mhealth-windows.npz'
    assert sha256(derived.read_bytes()).hexdigest() in (folder/'source.txt').read_text()
    with np.load(derived,allow_pickle=False) as data:
        ids = data['ids']
        assert data['windows'].shape == (3355,2,len(FEATURES))
    for person in range(1,11):
        assert (np.diff(np.sort(ids[ids[:,0]==person,2])) >= 100).all()
    subject_rows = pd.read_csv(folder/'by-subject.csv')
    with np.load(folder/'records.npz',allow_pickle=False) as archive:
        dev, test = archive['dev'],archive['test']
        assert set(ids[test,0]) == set(range(6,11)) and set(ids[dev,0]) == set(range(1,6))
        for part in ['train',*[f'p{i}' for i in range(6,11)]]:
            train = part == 'train'
            ref, queries = archive[f'{part}_ref'],archive[f'{part}_queries']
            assert not np.intersect1d(ref,queries).size
            q,h,m,device = archive[f'{part}_comparisons'].T
            assert np.isin(q,queries).all() and np.isin(np.r_[h,m],ref).all()
            assert (ids[h,1] == ids[q,1]).all() and (ids[m,1] != ids[q,1]).all()
            if train:
                assert np.isin(np.r_[q,h,m],dev).all()
                assert (ids[q,0] != ids[h,0]).all() and (ids[q,0] != ids[m,0]).all()
                assert len(q)==192 and device.sum()==128
                check_rates(archive['train_votes'],archive['rates'])
            else:
                assert set(ids[np.r_[q,h,m],0]) == {int(part[1:])}
        for r in subject_rows.itertuples():
            idx = ids[test,0] == r.subject
            yy = ids[test,1][idx]
            ba = [balanced_accuracy_score(yy,pred[idx]) for pred in archive[f'{r.method}_predictions']]
            np.testing.assert_allclose([*ba,min(ba)],[r.wrist_ba,r.ankle_ba,r.lower_ba],atol=1e-12)
    return len(subject_rows)


if __name__ == '__main__':
    base = ROOT/'results/minpi-study'
    root = base/'mechanism'
    total = simulated(root)+simulated(root/'calibration')
    for i,policy in enumerate((.5,2/3,5/6,1.)):
        total += simulated(root/'acquisition-bias',f'policy-{i}.npz',policy)
    print(f'{total} simulated support/endpoint records and {device(base/"device")} device records verified.')
