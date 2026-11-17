"""Compare the deployed ranker with the historical native path and archived wide paths."""
import ctypes
import hashlib
import json
import os
from threading import Timer

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_rework.predictive_polynomial_timing import OUT, wide
from experiments.minpi_rework import predictive_veto_timing as history
from experiments.minpi_compression import legacy_gain
from minppi_mofs.size_extrema import rank_minimum


def check():
    jobs = [(src,name,seed,"historical") for src in (*history.SOURCES,history.validation_predictive)
            for name in src.SOURCES for seed in range(42,47)]
    prepared = legacy_gain.jobs()
    jobs += [(None,name,seed,"prepared") for name,seed in sorted(prepared)]
    jobs += [(wide.data,name,seed,"wide") for name in wide.data.SOURCES for seed in range(42,47)]
    records = []
    for src,name,seed,panel in jobs:
        alarm = Timer(3600,lambda:os._exit(124))
        alarm.daemon = True
        alarm.start()
        try:
            if src is None:
                directory,pin = prepared[name,seed]
                x,y,_,idx = legacy_gain.read_split(directory,pin,seed)
                task = "regression" if pin["task"]=="regression" else "classification"
            else:
                x,y,task,idx = src.split(name,seed)
            model = wide.rf.fit(x[idx["fit"]],y[idx["fit"]],task,seed)
            raw = model.game(x[idx["vote"]])
            game = wide.pq.PredictiveGame.from_paths(raw.a,raw.incidence,wide.timing.centered(model,y[idx["vote"]],task))
            order,queries = rank_minimum(game)
            if panel=="wide":
                file = wide.ROOT/f"{name}-{seed}"/"selection.json"
                lock = json.loads((wide.OUT/"lock.json").read_bytes())["receipts"]
                assert wide.b.digest(file)==lock[file.parent.name]
                expected = json.loads(file.read_bytes())["path"]
            else:
                expected = game.ranked_path()[0]
            np.testing.assert_array_equal(order,expected)
            records.append(dict(panel=panel,dataset=name,seed=seed,p=len(order),checks=queries,
                                sha256=hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()))
        finally:
            alarm.cancel()
    pd.DataFrame(records).to_csv(OUT/"production-paths.csv",index=False)
    print(pd.Series([r["panel"] for r in records]).value_counts())


if __name__=="__main__":
    if os.name=="nt" and not ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<4)):
        raise OSError("Affinity failed")
    with threadpool_limits(1):
        check()
