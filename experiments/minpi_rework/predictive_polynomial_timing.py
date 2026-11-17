"""Warm, paired complete path timings on the four locked wide sources."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter
import zlib

import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits
from scipy import sparse
from experiments.minpi_rework import wide_veto_run as wide
from minppi_mofs.size_extrema import difference_tree, select


OUT = wide.b.ROOT/"thesis-paper/evidence/minpi_rework/predictive-polynomial"
METHODS = ("coalesced_early", "coalesced_straight", "coalesced_blas", "interval")


@njit(cache=True)
def rank_interval(initial,linear,ptr,idx,gv,ip,ii,jp,ji,unions,w,tol):
    low, high, absmax = difference_tree(w)
    live = np.diff(jp)
    keep = np.ones(len(live), np.bool_)
    alive = np.ones(len(ip)-1, np.bool_)
    delta = initial.copy()
    current = np.zeros(initial.shape[1])
    order = np.empty(len(keep), np.int64)
    t = count = tests = 0
    last = len(w)-1
    while t < len(keep):
        mask = wide._representatives(keep,live)
        j,last,nq,ni = select(current,delta,mask,w,tol,last,low,high,absmax)
        count += nq
        tests += ni
        t = wide._step(j,True,live,keep,order,t,alive,delta,current,linear,ptr,idx,gv,ip,ii,jp,ji,unions)
    return order,count,tests


def warm():
    rng = np.random.default_rng(573)
    inc = sparse.csr_matrix([[1,0,0,0,0],[1,0,1,0,0],[0,0,1,0,1]])
    game = wide.pq.PredictiveGame.from_paths(sparse.csc_matrix(rng.normal(size=(8,3))),inc,rng.normal(size=8))
    args = wide.args_for(game)
    order = rank_interval(*args)[0]
    for method in (*METHODS[:-1], "straight"):
        np.testing.assert_array_equal(order, wide.rank(method,args)[0])


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    warm()
    lock = json.loads((wide.OUT/"lock.json").read_bytes())["receipts"]
    rows = []
    for name in ("tehran","uji","oraltox","adverts"):
        methods = (*METHODS, "straight") if name == "tehran" else METHODS
        for seed in range(42,47):
            timer = Timer(3600,lambda:os._exit(124))
            timer.daemon = True
            timer.start()
            started = perf_counter()
            try:
                key = f"{name}-{seed}"
                receipt = wide.ROOT/key/"selection.json"
                assert wide.b.digest(receipt) == lock[key]
                rec = json.loads(receipt.read_bytes())
                x,y,task,idx = wide.data.split(name,seed)
                forest = wide.rf.fit(x[idx["fit"]],y[idx["fit"]],task,seed)
                raw = forest.game(x[idx["vote"]])
                centered = wide.timing.centered(forest,y[idx["vote"]],task)
                game = wide.pq.PredictiveGame.from_paths(raw.a,raw.incidence,centered)
                args = wide.args_for(game)
                rng = np.random.default_rng(zlib.crc32(key.encode())+12993)
                others = sum(rec[f] for f in ("fit_s","prepare_s","acceptance_s","pred_s"))
                for repeat in range(3):
                    for method in rng.permutation(methods):
                        tick = perf_counter()
                        if method == "interval":
                            order, queries, intervals = rank_interval(*args)
                        else:
                            order, queries = wide.rank(method,args)
                            intervals = 0
                        sec = perf_counter()-tick
                        np.testing.assert_array_equal(order, rec["path"])
                        rows.append(dict(dataset=name,seed=seed,p=raw.p,method=str(method),repeat=repeat,
                                         rank_s=sec,checks=queries,intervals=intervals,common_s=others,
                                         path_sha256=hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()))
                pd.DataFrame(rows).to_csv(OUT/"repeats.csv",index=False)
                print(key,f"{perf_counter()-started:.1f}s",flush=True)
            finally:
                timer.cancel()


if __name__ == "__main__":
    if os.name == "nt" and not ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<10)):
        raise OSError("Affinity failed")
    with threadpool_limits(1):
        collect()
