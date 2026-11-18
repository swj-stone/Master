"""Process-isolated coalition refits with a durable common fit ledger."""
from __future__ import annotations

from collections import deque
import ctypes
import faulthandler
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
from threading import Thread, get_native_id
import time
import traceback


class _WindowsJob:
    """Close our sole job handle on process termination to kill all attached children."""
    def __init__(self):
        self.handle=None
        if os.name!="nt":
            return
        from ctypes import wintypes as wt
        class Basic(ctypes.Structure):
            _fields_=[("process_time",ctypes.c_int64),("job_time",ctypes.c_int64),("flags",wt.DWORD),
                      ("minimum_ws",ctypes.c_size_t),("maximum_ws",ctypes.c_size_t),("process_limit",wt.DWORD),
                      ("affinity",ctypes.c_size_t),("priority",wt.DWORD),("scheduling",wt.DWORD)]
        class Limits(ctypes.Structure):
            _fields_=[("basic",Basic),("io",ctypes.c_uint64*6),("memory",ctypes.c_size_t*4)]
        self.kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes=(ctypes.c_void_p,wt.LPCWSTR)
        self.kernel.CreateJobObjectW.restype=wt.HANDLE
        self.kernel.SetInformationJobObject.argtypes=(wt.HANDLE,ctypes.c_int,ctypes.c_void_p,wt.DWORD)
        self.kernel.AssignProcessToJobObject.argtypes=(wt.HANDLE,wt.HANDLE)
        self.kernel.CloseHandle.argtypes=(wt.HANDLE,)
        job=self.kernel.CreateJobObjectW(None,None)
        if job:
            limits=Limits()
            limits.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if self.kernel.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)):
                self.handle=job
            else:
                self.kernel.CloseHandle(job)

    def attach(self,process):
        return bool(self.handle and self.kernel.AssignProcessToJobObject(self.handle,process.sentinel))

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle=None


def _watch_parent():
    parent=mp.parent_process()
    wait([parent.sentinel])
    os._exit(77)


def _serve(pipe,slot,source,rank_rows,policy,method,utility,folder,started,lock,counters,active,methods):
    # This fallback also runs on systems that disallow attaching to a nested job.
    Thread(target=_watch_parent,daemon=True,name="coordinator-lifetime").start()
    crash=(folder/f"crash-fit-{os.getpid()}.log").open("a",buffering=1,encoding="utf-8")
    faulthandler.enable(file=crash,all_threads=True)
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        from experiments.literature_worker import _loss
        from experiments.test.piv3_lite_arm import _predict
        from minppi_mofs.modeling import make_model
        from minppi_mofs.utility import compute_utility
        def load(name):
            return np.load(source/f"{name}.npy",mmap_mode="r",allow_pickle=False)
        train=np.asarray(load("train"),dtype=np.float32,order="C")
        yt=load("train_y")
        indices=np.asarray(rank_rows,dtype=np.intp)
        rank=np.asarray(load("val")[indices],dtype=np.float32,order="C")
        yr=load("val_y")[indices]
        task=policy["task"]
        def event(kind,fit_id,mask,digest):
            row=dict(event=kind,fit_id=fit_id,phase="coalition",k=len(mask),elapsed_s=time.perf_counter()-started,
                     pid=os.getpid(),thread_id=get_native_id(),prefix_sha256=digest)
            if methods is not None:
                row["methods"]=methods
            with (folder/"fits.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(row)+"\n")
        with threadpool_limits(limits=1):
            pipe.send(("ready",os.getpid()))
            while True:
                item=pipe.recv()
                if item is None:
                    break
                index,mask=item
                x=np.take(train,mask,axis=1)
                model=make_model(**policy)
                digest=hashlib.sha256(np.asarray(mask,dtype="<u4").tobytes()).hexdigest()
                with lock:
                    counters[0]+=1
                    fit_id=counters[0]
                    active[slot]=fit_id
                    counters[2]=max(counters[2],sum(bool(i) for i in active))
                    event("started",fit_id,mask,digest)
                try:
                    model.fit(x,yt)
                except BaseException:
                    with lock:
                        active[slot]=0
                        event("failed",fit_id,mask,digest)
                    raise
                with lock:
                    counters[1]+=1
                    active[slot]=0
                    event("completed",fit_id,mask,digest)
                del x
                pred=_predict(model,np.take(rank,mask,axis=1),task=="regression")
                del model
                if method=="minshap":
                    value=_loss(yr,pred,task)
                else:
                    value=float(compute_utility(yr,pred,metric=utility))
                    if method=="permutation_pair":
                        value=_loss(yr,pred,task),value
                pipe.send(("value",index,value))
    except BaseException:
        details=traceback.format_exc()
        print(details,file=crash,flush=True)
        try:
            pipe.send(("error",details))
        except (OSError,EOFError):
            pass
        raise
    finally:
        pipe.close()
        faulthandler.disable()
        crash.close()


class NativeFitPool:
    def __init__(self,*,source,rank_rows,policy,method,utility,ledger,workers=2):
        if workers!=2:
            raise ValueError("The isolated native path is designed for two workers")
        self.ledger=ledger
        self.ctx=mp.get_context("spawn")
        self.lock=self.ctx.Lock()
        self.counters=self.ctx.RawArray("q",[ledger.initiated,ledger.finished,ledger.peak_inflight])
        self.active=self.ctx.RawArray("q",workers)
        self.args=(Path(source),rank_rows,policy,method,utility,ledger.folder,ledger.started,self.lock,
                   self.counters,self.active,ledger.fit_users)
        self.children=[]
        self.connections=[]
        self.job=_WindowsJob()
        self.job_attached=[]

    @property
    def pids(self):
        return [p.pid for p in self.children]

    def __enter__(self):
        try:
            for slot in range(len(self.active)):
                coordinator,worker=self.ctx.Pipe()
                proc=self.ctx.Process(target=_serve,args=(worker,slot,*self.args),daemon=True)
                try:
                    proc.start()
                finally:
                    worker.close()
                self.children.append(proc)
                self.connections.append(coordinator)
                self.job_attached.append(self.job.attach(proc))
            from minppi_mofs.experiment_protocol import atomic_write_json
            atomic_write_json(self.ledger.folder/"fit_processes.json",
                              dict(coordinator_pid=os.getpid(),pids=self.pids,windows_job=self.job_attached,
                                   fallback="multiprocessing parent sentinel"))
            for conn in self.connections:
                self._recv(conn,"ready")
        except BaseException:
            self.__exit__(True,None,None)
            raise
        return self

    def _recv(self,conn,kind):
        try:
            msg=conn.recv()
        except (EOFError,OSError) as error:
            raise RuntimeError("Isolated refit worker exited; inspect crash-fit-*.log and fits.jsonl") from error
        if msg[0]=="error":
            raise RuntimeError(msg[1])
        if msg[0]!=kind:
            raise RuntimeError(f"Unexpected native worker response {msg[0]}")
        return msg

    def _sync(self):
        with self.lock:
            self.ledger.initiated,self.ledger.finished,self.ledger.peak_inflight=self.counters[:]
            self.ledger.inflight={i for i in self.active if i}
        self.ledger.progress(self.ledger.stage)

    def values(self,masks):
        results=[None]*len(masks)
        todo=deque(enumerate(masks))
        active=[]
        for conn in self.connections:
            if todo:
                conn.send(todo.popleft())
                active.append(conn)
        while active:
            for conn in wait(active,timeout=.5):
                _,i,results_i=self._recv(conn,"value")
                results[i]=results_i
                if todo:
                    conn.send(todo.popleft())
                else:
                    active.remove(conn)
            self._sync()
        return results

    def __exit__(self,exc_type,exc,trace):
        try:
            if not exc_type:
                for conn in self.connections:
                    conn.send(None)
            else:
                for p in self.children:
                    if p.is_alive():
                        p.kill()
            for p in self.children:
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
                if not exc_type and p.exitcode!=0:
                    raise RuntimeError(f"Isolated refit worker {p.pid} exited with code {p.exitcode}")
        finally:
            self.job.close()
            for conn in self.connections:
                conn.close()
            for p in self.children:
                if p.is_alive():
                    p.kill()
                p.join()
                p.close()
