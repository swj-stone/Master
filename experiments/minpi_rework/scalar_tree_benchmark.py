"""Paired native runtimes for scalar coalition votes on sealed confirmation paths."""
import ctypes
from hashlib import sha256
import json
import os
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import legacy_gain as legacy
from experiments.minpi_rework import quota_compress_tree as tree
from experiments.minpi_rework import quota_compress_tree_confirmation as confirm
from experiments.minpi_rework.quota_compress_tree_native_exact import direct
from experiments.minpi_rework.scalar_tree_ballots import scalar
from minppi_mofs.modeling import make_model


ARMS = ("direct", "direct_block32", "sizes", "dynamic", "sizes_dynamic")


def run(game, order, mode, rho=.2):
    a, inc = game.a, game.incidence
    by = inc.tocsc()
    p = len(order)
    limit = rho*game.norm + 1e-12*max(1.,game.norm)
    args = (len(game.full),a.indptr,a.indices,a.data,inc.indptr,inc.indices,
            by.indptr,by.indices,order,limit)
    if mode in ("direct","direct_block32"):
        path, calls = direct(*args,mode=="direct_block32")
        return path, calls, 0
    sizes = np.unique([0,(p-1)//2,p-1]) if mode != "dynamic" else np.empty(0,dtype=np.int64)
    return scalar(*args,sizes,mode!="sizes")


def check():
    from itertools import combinations
    rng = np.random.default_rng(3901)
    checked = 0
    for p in range(1,8):
        groups = [g for k in range(1,min(4,p)+1) for g in combinations(range(p),k)]
        inc = csr_matrix([[int(j in g) for j in range(p)] for g in groups])
        for trial in range(3):
            parts = rng.integers(-3,4,size=(8,len(groups))).astype(float)
            if trial == 0:
                parts[:] = 0
            g = tree.Game(csc_matrix(parts),inc)
            order = rng.permutation(p)
            for rho in (0.,.2,.8,1.):
                ref = run(g,order,"direct",rho)[0]
                for mode in ARMS[1:]:
                    np.testing.assert_array_equal(run(g,order,mode,rho)[0],ref)
                checked += 1
    return checked


def summarize():
    daily = pd.read_csv(confirm.DEST/"scalar-ballots.csv")
    key = ["dataset","seed"]
    assert daily.groupby(key).path_sha256.nunique().eq(1).all()
    timing = daily.groupby([*key,"method"]).median(numeric_only=True).unstack()
    rng = np.random.default_rng(81191)
    draws = rng.integers(0,13,size=(10000,13))
    rows = []
    for method in ("sizes","dynamic","sizes_dynamic"):
        for baseline in ("direct","direct_block32","dynamic"):
            if method == baseline:
                continue
            ratio = np.log(timing.search_s[method]/timing.search_s[baseline]).groupby(level=0).mean()
            assert len(ratio)==13
            boots = np.exp(ratio.to_numpy()[draws].mean(axis=1))
            low, high = np.quantile(boots,[.025,.975])
            rows.append(dict(method=method,baseline=baseline,search_ratio=np.exp(ratio.mean()),
                lower95=low,upper95=high,sources_faster=int((ratio<0).sum()),
                scan_fraction=timing.candidate_scans[method].sum()/timing.candidate_scans[baseline].sum()))
    result = pd.DataFrame(rows)
    result.to_csv(confirm.DEST/"scalar-ballot-contrasts.csv",index=False)
    print(result.round(3).to_string(index=False))


def main():
    if os.name == "nt":
        dll = ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<3)):
            raise ctypes.WinError(ctypes.get_last_error())
    print(f"Matched {check()} finite test games across five native arms.",flush=True)
    rows = []
    started = perf_counter()
    lock = json.loads((confirm.DEST/"lock.json").read_bytes())["receipts"]
    for k,(panel,source,name,seed) in enumerate(confirm.jobs(),1):
        folder = confirm.OUT/f"{name}-{seed}"
        saved = (folder/"selection.json").read_bytes()
        assert sha256(saved).hexdigest() == lock[folder.name]
        target = json.loads(saved)
        x,y,task,idx = source.split(name,seed)
        pool = np.sort(np.r_[idx["fit"],idx["vote"]])
        problem = ("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
        model = make_model(task=problem,seed=seed,backend=legacy.CONFIG["backend"],
                           model_params=legacy.CONFIG["model_params"])
        model.fit(x[pool],y[pool])
        val = x[idx["width"]]
        game = tree.extract(model,val[:1]).cache(val)
        gain = model.get_booster().get_score(importance_type="gain")
        order = np.array(sorted(range(game.p),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
        rng = np.random.default_rng(seed)
        for rep in range(-1,5):
            for mode in rng.permutation(ARMS):
                t = perf_counter()
                path, calls, vetoes = run(game,order,mode)
                seconds = perf_counter()-t
                np.testing.assert_array_equal(path,target["path"])
                if rep >= 0:
                    digest = sha256(np.asarray(path,dtype="<i4").tobytes()).hexdigest()
                    rows.append(dict(panel=panel,dataset=name,seed=seed,method=mode,repeat=rep,
                        search_s=seconds,candidate_scans=calls,vetoes=vetoes,path_sha256=digest))
        pd.DataFrame(rows).to_csv(confirm.DEST/"scalar-ballots.csv",index=False)
        elapsed = perf_counter()-started
        print(f"[{k}/65] {name}/{seed} equal path, elapsed {elapsed:.1f}s ETA {elapsed*(65-k)/k:.1f}s",
              flush=True)
    summarize()


if __name__=="__main__":
    with threadpool_limits(1):
        main()
