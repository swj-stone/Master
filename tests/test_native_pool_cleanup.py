"""A hard-stopped experiment coordinator must leave no native workers behind."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest


def coordinate(folder):
    from experiments.literature_worker import _Ledger
    from experiments.native_fit_pool import NativeFitPool

    policy=dict(task="binary",model_params={"max_depth":1,"n_estimators":1000,"n_jobs":1},
                backend="cpu",seed=42)
    ledger=_Ledger(folder,time.perf_counter(),"synthetic","native","shapley",native_workers=2)
    with NativeFitPool(source=folder,rank_rows=tuple(range(16)),policy=policy,
                       method="shapley",utility="roc_auc",ledger=ledger,workers=2) as pool:
        pids=list(pool.pids)
        temporary=folder/"pids.tmp"
        temporary.write_text(json.dumps(pids),encoding="utf-8")
        temporary.replace(folder/"pids.json")
        # The controlling test terminates this process before normal context cleanup.
        pool.values([(0,), (1,)])
        time.sleep(60)


@pytest.mark.skipif(os.name!="nt",reason="Checks the Windows benchmark watchdog kill path")
def test_hard_kill_of_coordinator_reaps_both_native_workers(tmp_path):
    import _winapi
    repo=Path(__file__).resolve().parents[1]
    for name in ("train","val"):
        np.save(tmp_path/(name+".npy"),np.arange(48,dtype="float32").reshape(16,3))
        np.save(tmp_path/(name+"_y.npy"),np.arange(16)%2)
    env=dict(os.environ)
    env["PYTHONPATH"]=os.pathsep.join([str(repo),str(repo/"src"),env.get("PYTHONPATH","")])
    env.update(OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1")
    log=tmp_path/"coordinator.log"
    children=[]
    with log.open("w") as out:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(tmp_path)],cwd=repo,
                              env=env,stdout=out,stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            limit=time.monotonic()+45
            while (not (tmp_path/"pids.json").is_file() and proc.poll() is None
                   and time.monotonic()<limit):
                time.sleep(.05)
            assert (tmp_path/"pids.json").is_file(), log.read_text(encoding="utf-8",errors="replace")
            pids=json.loads((tmp_path/"pids.json").read_bytes())
            assert len(set(pids))==2
            process_terminate=0x0001
            children=[_winapi.OpenProcess(_winapi.SYNCHRONIZE|process_terminate,False,pid) for pid in pids]
            while not (tmp_path/"fits.jsonl").is_file() and time.monotonic()<limit:
                time.sleep(.01)
            assert (tmp_path/"fits.jsonl").is_file(), "No physical native fit recorded"
            assert all(_winapi.WaitForSingleObject(handle,0)==_winapi.WAIT_TIMEOUT for handle in children)
            proc.kill()
            proc.wait(timeout=10)
            assert all(_winapi.WaitForSingleObject(handle,10000)==0 for handle in children), pids
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            for handle in children:
                if _winapi.WaitForSingleObject(handle,0)==_winapi.WAIT_TIMEOUT:
                    _winapi.TerminateProcess(handle,1)
                    _winapi.WaitForSingleObject(handle,5000)
                _winapi.CloseHandle(handle)


if __name__=="__main__":
    coordinate(Path(sys.argv[1]))
