"""Standalone timings of guarded width interpolation. No test labels enter selection."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.anchored_vote.audit import score
from experiments.minpi_design.study import load
from experiments.minpi_replication.direct_gain import fit_native, gain_order, predict_native, validation_utility
from experiments.split_vote.run import ORIGINAL, ROOT
from experiments.vote_increment.audit import checks
from experiments.vote_increment.path import search as game


def select(x, y, task, idx, seed, arm):
    tick = perf_counter()
    tr = np.sort(idx['fit'])
    va = np.sort(np.r_[idx['vote'], idx['width']])
    dev = np.sort(np.r_[tr, va])
    p = x.shape[1]
    model = fit_native(x[tr], y[tr], task, seed)
    order = gain_order(model, p)
    profile = game.loss_profile(model, order, arm)

    def measure(k):
        cols = sorted(order[:k])
        check = model if k == p else fit_native(x[np.ix_(tr,cols)], y[tr], task, seed)
        pred = predict_native(check, x[np.ix_(va,cols)], task)
        return validation_utility(y[va], pred, task)

    k, trace = game.search(profile, measure)
    cols = sorted(order[:k])
    final = fit_native(x[np.ix_(dev,cols)], y[dev], task, seed)
    pred = predict_native(final, x[np.ix_(idx['test'],cols)], task)
    return dict(k=k,selected=cols,trace=trace,fits=len(trace)+1,seconds=perf_counter()-tick), pred


def run(args):
    root = ROOT/'results/vote_increment/path'/args.tag
    src = sha256(Path(__file__).read_bytes()+Path(game.__file__).read_bytes()).hexdigest()
    rows = []
    for j, seed in enumerate(args.seeds):
        for i, name in enumerate(args.names):
            x,y,task,idx,labels = load(name,seed)
            x = np.asarray(x, dtype=np.float32)
            arms = np.roll(['gain','mean','min'], i+j)
            folder = root/f'{name}-{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            for arm in arms:
                receipt = folder/f'{arm}.json'
                path = folder/f'{arm}.npz'
                if not receipt.exists():
                    rec, pred = select(x,y,task,idx,seed,arm)
                    np.savez_compressed(path,rows=idx['test'],pred=pred)
                    rec.update(dataset=name,seed=seed,method=str(arm),source=src,forecast=sha256(path.read_bytes()).hexdigest())
                    receipt.write_text(json.dumps(rec,allow_nan=False),encoding='utf8')
                    print(name,seed,arm,rec['k'],rec['fits'],round(rec['seconds'],2),flush=True)
            truth = labels()
            for arm in arms:
                path = folder/f'{arm}.npz'
                rec = json.loads((folder/f'{arm}.json').read_bytes())
                if rec['source']!=src or rec['forecast']!=sha256(path.read_bytes()).hexdigest():
                    raise ValueError(f'Stale {folder}/{arm}')
                with np.load(path) as saved:
                    np.testing.assert_array_equal(saved['rows'],idx['test'])
                    metric = score(truth,saved['pred'],task)
                rows.append({k:v for k,v in rec.items() if k not in ('source','forecast')}|metric)
    frame = pd.DataFrame(rows)
    dest = ROOT/'thesis-paper/evidence/vote_increment/path'/args.tag
    dest.mkdir(parents=True,exist_ok=True)
    frame.to_csv(dest/'results.csv',index=False)
    paired,sources = checks(frame,args.names,args.seeds)
    paired.to_csv(dest/'gate-seeds.csv',index=False)
    sources.to_csv(dest/'gate-sources.csv',index=False)
    print(sources.to_string(index=False), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--names',nargs='+',default=list(ORIGINAL))
    parser.add_argument('--seeds',type=int,nargs='+',default=[42])
    with threadpool_limits(1):
        run(parser.parse_args())
