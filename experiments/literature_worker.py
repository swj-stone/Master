"""From-scratch feature selection on a dataset partition prepared for the requested seed."""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from threading import RLock, get_native_id

import numpy as np
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from experiments.test.piv3_lite_arm import _matrix, _metrics, _predict
from experiments.test.piv3_raw_splits import DEFAULT_ROOT, load_split
from minppi_mofs.baselines import get_selector
from minppi_mofs.baselines._selection_context import Selection, SelectionContext
from minppi_mofs.baselines.pfi import sampled_rows
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash
from minppi_mofs.modeling import make_model
from minppi_mofs.piv3_compact_tree import choose_prefix
from minppi_mofs.prediction_meter import PredictionMeter, minpi_sample_counts
from minppi_mofs.utility import compute_utility


FROZEN_ROWS = 256
METHODS = ("minshap", "shapley", "banzhaf", "shapg")
CONTROLS = ("minpi", "gain", "pfi", "full", "sampling_shap")


def _constant(y, n, regression):
    if regression:
        return np.full(n, np.mean(y), dtype=float)
    counts = np.bincount(y)
    return np.broadcast_to(counts / len(y), (n, len(counts)))


def _loss(y, pred, task):
    if task == "regression":
        return np.square(y-pred)
    if task == "binary":
        return np.square(y-pred[:,1])
    return np.square(np.eye(pred.shape[1])[y]-pred).mean(axis=1)


def _failures(reference, candidate):
    return [k for k,b in reference.items() if
            ((candidate[k]-b) if k == "mae" else (b-candidate[k])) > .05*abs(b)+1e-12]


def _subset(frame, indices):
    if len(indices) == frame.shape[1]:
        return frame
    return frame.iloc[:,list(indices)]


class _Ledger:
    def __init__(self, folder, started, dataset, mode, method, native_workers=1):
        self.folder = folder
        self.started = started
        self.identity = dict(dataset=dataset,mode=mode,method=method)
        self.initiated = self.finished = self.constant_calls = 0
        self.last_tick = -float("inf")
        self.stage = ""
        self.fields = {}
        self.coalitions = self.unique_calls = self.pred_rows = 0
        self.fit_users = None
        self.lock = RLock()
        self.native_workers = native_workers
        self.inflight = set()
        self.peak_inflight = 0

    def _event(self, event, phase, k, fit_id, prefix_hash):
        with (self.folder/"fits.jsonl").open("a",encoding="utf-8") as stream:
            record=dict(event=event,fit_id=fit_id,phase=phase,k=k,elapsed_s=time.perf_counter()-self.started)
            if self.fit_users is not None:
                record["methods"]=self.fit_users
            if self.native_workers>1:
                record.update(pid=os.getpid(),thread_id=get_native_id())
                if prefix_hash is not None:
                    record["prefix_sha256"]=prefix_hash
            stream.write(json.dumps(record)+"\n")

    def progress(self, phase, **fields):
        with self.lock:
            self._progress(phase,**fields)

    def _progress(self, phase, **fields):
        now = time.perf_counter()
        changed = phase != self.stage
        self.fields = fields if changed else dict(self.fields, **fields)
        self.stage = phase
        if changed and self.native_workers>1:
            with (self.folder/"stages.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(dict(phase=phase,thread_id=get_native_id(),fits=self.finished,
                    elapsed_s=now-self.started,inflight_fit_ids=sorted(self.inflight),**fields))+"\n")
        if changed or now-self.last_tick >= .5:
            self.last_tick = now
            atomic_write_json(self.folder/"progress.json",dict(
                **self.identity,phase=phase,elapsed_s=now-self.started,fits=self.finished,
                fits_started=self.initiated,coalition_calls=self.coalitions,
                actual_coalitions=self.unique_calls,ranking_prediction_rows=self.pred_rows,
                native_workers=self.native_workers,inflight_fit_ids=sorted(self.inflight),
                peak_inflight_fits=self.peak_inflight,
                **self.fields))

    def fit(self, policy, x, y, phase, *, mask=None):
        model = make_model(**policy)
        prefix_hash=(hashlib.sha256(np.asarray(mask,dtype="<u4").tobytes()).hexdigest()
                     if self.native_workers>1 and mask is not None else None)
        with self.lock:
            self.initiated += 1
            fit_id=self.initiated
            self.inflight.add(fit_id)
            self.peak_inflight=max(self.peak_inflight,len(self.inflight))
            self._event("started",phase,x.shape[1],fit_id,prefix_hash)
            if self.native_workers>1:
                self._progress(self.stage)
        try:
            model.fit(x,y)
        except BaseException:
            with self.lock:
                self.inflight.remove(fit_id)
                self._event("failed",phase,x.shape[1],fit_id,prefix_hash)
            raise
        with self.lock:
            self.finished += 1
            self.inflight.remove(fit_id)
            self._event("completed",phase,x.shape[1],fit_id,prefix_hash)
        return model


class _Oracle:
    def __init__(self, train, yt, rank, yr, mode, method, full, policy, utility, ledger):
        self.train, self.yt, self.rank, self.yr = train, yt, rank, np.asarray(yr)
        self.mode, self.method = mode, method
        self.full, self.policy, self.utility = full, policy, utility
        self.task, self.width = policy["task"], train.shape[1]
        self.reg = self.task == "regression"
        self.ledger = ledger
        self.owner_thread = get_native_id()
        self.full_indices = tuple(range(self.width))
        self.cache = {}
        self.permutation = method in ("minshap","shapley","permutation_pair")
        self.n_cache_hits = 0
        self.n_predictions = 0
        self.active = tuple(sorted(int(f[1:]) for f in full.get_booster().get_score(importance_type="weight")))
        self.use = set(self.active)
        if mode == "frozen":
            self.rows = sampled_rows(self.yr,FROZEN_ROWS,self.reg,policy["seed"])
            self.x = _matrix(rank.iloc[self.rows])
            self.labels = self.yr[self.rows]
            self.weights = np.ones(len(self.rows),dtype=float)
            if not self.reg:
                _, counts = np.unique(self.yr,return_counts=True)
                seen = np.bincount(self.labels, minlength=len(counts))
                self.weights = (counts / seen)[self.labels]
            tr = np.random.default_rng(policy["seed"]).choice(len(train),min(512,len(train)),replace=False)
            self.fill = np.nanmedian(_matrix(train.iloc[tr]), axis=0)
            self.predictor=None
            if self.permutation:
                from minppi_mofs.baselines._frozen_prediction import FrozenMaskPredictor
                self.predictor=FrozenMaskPredictor(full,self.x,self.fill,self.active,regression=self.reg)
        else:
            self.rows = np.arange(len(yr))
            if mode=="native" and self.permutation:
                self.training_matrix=_matrix(train)
                self.ranking_matrix=_matrix(rank)

    def native(self, mask, need_losses=False, joint=False):
        if self.mode != "native":
            raise ValueError("Refitted utility accessed on a different game")
        mask = tuple(mask) if self.permutation else tuple(sorted(mask))
        with self.ledger.lock:
            self.ledger.coalitions += 1
        if mask in self.cache:
            self.n_cache_hits += 1
            return self.cache[mask]
        if not mask:
            pred = _constant(self.yt,len(self.yr),self.reg)
            self.ledger.constant_calls += 1
        else:
            if self.permutation:
                # Gather straight into C-contiguous matrices accepted by XGBoost.
                # Sorted coordinates preserve the original fit's feature order.
                is_full=len(mask)==self.width
                if is_full and get_native_id()!=self.owner_thread:
                    raise RuntimeError("The full-data booster must be evaluated by the thread that owns it")
                model=self.full if is_full else self.ledger.fit(
                    self.policy,np.take(self.training_matrix,mask,axis=1),self.yt,"coalition",mask=mask)
                data=self.ranking_matrix if is_full else np.take(self.ranking_matrix,mask,axis=1)
            else:
                model=self.full if mask==self.full_indices else self.ledger.fit(
                    self.policy,_matrix(_subset(self.train,mask)),self.yt,"coalition")
                data=_matrix(_subset(self.rank,mask))
            pred = _predict(model,data,self.reg)
            del model
        with self.ledger.lock:
            self.n_predictions += 1
            self.ledger.unique_calls += 1
            self.ledger.pred_rows += len(self.yr)
        if joint:
            v = _loss(self.yr,pred,self.task), float(compute_utility(self.yr,pred,metric=self.utility))
        elif need_losses:
            v = _loss(self.yr,pred,self.task)
        elif self.method == "shapg":
            v = float(r2_score(self.yr,pred)) if self.reg else float(
                f1_score(self.yr,pred.argmax(axis=1),average="binary" if self.task=="binary" else "macro",zero_division=0))
        else:
            v = float(compute_utility(self.yr,pred,metric=self.utility))
        # The permutation traversal caches and releases repeated prefixes itself.
        if not self.permutation:
            self.cache[mask] = v
        return v

    def losses(self, mask):
        if self.method != "minshap":
            raise ValueError("Per-row native losses are specific to MinShap")
        return self.native(mask,need_losses=True)

    def value(self, mask):
        return self.values([mask])[0]

    def joint_values(self,mask):
        return self.native(mask,joint=True)

    def isolated_batch(self,observe,masks,pool):
        """The coordinator owns the reference booster. No model crosses a process boundary."""
        partial=[mask for mask in masks if 0<len(mask)<self.width]
        values=iter(pool.values(partial))
        self.ledger.coalitions+=len(partial)
        self.n_predictions+=len(partial)
        self.ledger.unique_calls+=len(partial)
        self.ledger.pred_rows+=len(partial)*len(self.yr)
        return [next(values) if 0<len(mask)<self.width else observe(mask) for mask in masks]

    def values(self, masks):
        if self.mode == "native":
            return [self.native(mask) for mask in masks]
        if self.mode != "frozen":
            raise ValueError("Unexpected coalition query on a shared control")
        # A coordinate absent from every split cannot affect a frozen prediction.
        keys = ([tuple(mask) for mask in masks] if self.permutation else
                [tuple(sorted(i for i in mask if i in self.use)) for mask in masks])
        cache={} if self.permutation else self.cache
        self.ledger.coalitions += len(keys)
        missing = list(dict.fromkeys(key for key in keys if key not in cache))
        self.n_cache_hits += len(keys)-len(missing)
        block = self.predictor.batch_size if self.predictor else max(1,min(64,4_000_000 // self.x.size))
        for start in range(0,len(missing),block):
            chunk = missing[start:start+block]
            values=(self.predictor.utilities(chunk,self.labels,self.weights) if self.predictor else
                    self._full_width_utilities(chunk))
            for key,v in zip(chunk,values,strict=True):
                cache[key]=float(v)
            self.n_predictions += 1
            self.ledger.unique_calls += len(chunk)
            self.ledger.pred_rows += len(chunk)*len(self.labels)
        return [cache[key] for key in keys]

    def _full_width_utilities(self,chunk):
        cube=np.broadcast_to(self.fill,(len(chunk),*self.x.shape)).copy()
        for pos,mask in enumerate(chunk):
            if mask:
                ids=list(mask)
                cube[pos][:,ids]=self.x[:,ids]
        pred=_predict(self.full,cube.reshape(-1,self.width),self.reg)
        pred=pred.reshape((len(chunk),len(self.labels),*pred.shape[1:]))
        if self.reg:
            return [max(0.,float(r2_score(self.labels,p,sample_weight=self.weights))) for p in pred]
        if self.task=="binary":
            return [roc_auc_score(self.labels,p[:,1],sample_weight=self.weights) for p in pred]
        return [roc_auc_score(self.labels,p,multi_class="ovr",average="macro",sample_weight=self.weights) for p in pred]

    def describe(self):
        return dict(utility_rows=len(self.rows),effective_split_features=len(self.active),
                    coalition_calls=self.ledger.coalitions,actual_coalitions=self.ledger.unique_calls,
                    cached_calls=self.n_cache_hits,ranking_prediction_rows=self.ledger.pred_rows,
                    oracle_batches=self.n_predictions,constant_calls=self.ledger.constant_calls)


def _candidate(ans,ctx,oracle):
    """Turn a ranking or native support into a frozen-model proposal without using guard/test."""
    if not isinstance(ans,Selection):
        raise TypeError("Expected selector to return a Selection")
    info=dict(ans.diagnostics)
    columns=ctx.columns
    if ans.scores is not None:
        scores=np.asarray(ans.scores,dtype=np.float64)
        if scores.shape!=(len(columns),) or not np.isfinite(scores).all():
            raise ValueError("Scores must be finite and align with the original input dimensions")
        info["scores"]=scores.tolist()
    if ans.selected is None:
        if ans.scores is None:
            raise ValueError("A ranking or a native support is required")
        positions=np.argsort(-scores,kind="stable")
        if ctx.mode=="frozen" and oracle.active:
            positions=[i for i in positions if i in oracle.use]
        order=[columns[i] for i in positions]
        candidate,sizing=choose_prefix(ctx.full_model,ctx.train,ctx.ranking,ctx.yr,task=ctx.task,
                                       utility=ctx.utility,order=order,seed=ctx.seed)
        info.update(sizing=sizing,selection_rule="common frozen ranked-prefix sizing")
    else:
        candidate=list(ans.selected)
        info["selection_rule"]="intrinsic or published control selection"
    members=set(candidate)
    if len(members)!=len(candidate) or candidate!=[c for c in columns if c in members]:
        raise ValueError("Invalid, duplicate or noncanonical candidate features")
    return candidate,info


def _prediction_usage(method, info, meter):
    """Count predictor inputs during selection, excluding guard and final assessments."""
    measured = meter.counts
    estimate = measured["estimate"]
    if method == "minpi":
        phases = minpi_sample_counts(info, estimate)
        sampled = phases["voting"]
        policy = "three-context deletion voting on sampled ranking rows"
    else:
        if estimate != {"rows": info["prediction_rows"], "calls": info["prediction_calls"]}:
            raise ValueError("SamplingSHAP counters disagree with the full-model prediction meter")
        sampled = {"rows": info["sampling_prediction_rows"], "calls": info["sampling_prediction_calls"]}
        phases = {"explanation_including_reference": estimate, "sampled_pairs": sampled,
                  "prefix": measured["sizing"]}
        policy = "SamplingExplainer paired hybrid predictions"
    info["prediction_counts_by_phase"] = phases
    return dict(sample_prediction_rows=sampled["rows"], sample_prediction_calls=sampled["calls"],
                selection_prediction_rows=sum(v["rows"] for v in measured.values()),
                selection_prediction_calls=sum(v["calls"] for v in measured.values()), sampling_policy=policy)


def trial(dataset,mode,method,out,*,enforce_guard=True,minshap_support_only=False,native_workers=1,seed=42,input_root=None):
    """Run one cold, externally supervised route. Train and ranking data are the only selector inputs."""
    paired=method=="permutation_pair"
    methods=("minshap","shapley") if paired else (method,)
    if (mode=="shared" and method not in CONTROLS) or (mode in ("native","frozen") and method not in METHODS and not paired):
        raise ValueError(f"Unknown literature arm {mode}/{method}")
    if mode not in ("shared","native","frozen"):
        raise ValueError(f"Unknown game {mode}")
    if minshap_support_only and (mode,method)!=("native","minshap"):
        raise ValueError("Support-only traversal applies to standalone native MinShap")
    if native_workers not in (1,2) or (native_workers>1 and (mode!="native" or method not in ("minshap","shapley","permutation_pair"))):
        raise ValueError("Two workers are available only for native permutation games")
    if type(seed) is not int or not 0<=seed<=np.iinfo(np.uint32).max:
        raise ValueError("Expected an integer random seed in the uint32 range")
    started=time.perf_counter()
    source_root=DEFAULT_ROOT if input_root is None else Path(input_root)
    source=source_root/dataset
    meta=json.loads((source/"metadata.json").read_bytes())
    policy={k:v for k,v in meta["policy"].items() if k!="utility"}
    if meta.get("dataset",dataset)!=dataset or meta.get("seed",policy["seed"])!=seed or policy["seed"]!=seed:
        raise ValueError(f"Input partition or model policy does not match {dataset}/seed-{seed}")
    out=Path(out)
    ledger=_Ledger(out,started,dataset,mode,method,native_workers)
    ledger.identity["seed"]=seed
    if paired:
        ledger.fit_users=methods
    folders={m:out/m if paired else out for m in methods}
    for folder in folders.values():
        folder.mkdir(exist_ok=True)
    progress=ledger.progress
    progress("loading")
    task=policy["task"]
    regression=task=="regression"
    utility=meta["utility"]
    fraud=dataset=="credit_card_fraud"
    train,yt=load_split(source_root,dataset,"train")
    val,yv=load_split(source_root,dataset,"val")
    columns=list(train.columns)
    if list(val.columns)!=columns or columns!=meta["features"] or not train.columns.is_unique:
        raise ValueError("Feature schema changed")
    rank_rows,guard_rows=train_test_split(np.arange(len(val)),test_size=.5,random_state=seed,
                                         stratify=None if regression else yv)
    rank_rows,guard_rows=np.sort(rank_rows),np.sort(guard_rows)
    rank,yr=val.iloc[rank_rows],np.asarray(yv)[rank_rows]
    guard,yg=val.iloc[guard_rows],np.asarray(yv)[guard_rows]
    score=lambda labels,pred: _metrics(labels,pred,utility,fraud)

    progress("reference")
    tick=time.perf_counter()
    full=ledger.fit(policy,_matrix(train),yt,"reference")
    full_guard=_predict(full,_matrix(guard),regression)
    ref=score(yg,full_guard)
    reference_s=time.perf_counter()-tick
    progress("selection")
    tick=time.perf_counter()
    selector_begin=tick
    oracle=_Oracle(train,yt,rank,yr,mode,method,full,policy,utility,ledger)
    meter = PredictionMeter(full) if method in ("minpi", "sampling_shap") else None
    ctx=SelectionContext(method=method,mode=mode,columns=columns,train=train,yt=yt,ranking=rank,yr=yr,
                         task=task,utility=utility,seed=seed,full_model=meter if meter is not None else full,
                         active=oracle.active if mode=="frozen" else None,value=oracle.value,
                         values=oracle.values,losses=oracle.losses,progress=progress)
    ctx.minshap_support_only=minshap_support_only
    ctx.native_workers=native_workers
    pool_pids=[]
    if native_workers>1:
        from experiments.native_fit_pool import NativeFitPool
        executor=NativeFitPool(source=source,rank_rows=rank_rows,policy=policy,method=method,
                               utility=utility,ledger=ledger,workers=native_workers)
        progress("starting_fit_processes")
    else:
        executor=nullcontext()
    with executor as pool:
        if pool is not None:
            pool_pids=pool.pids
            progress("fit_processes_ready",pids=pool_pids)
            observe=oracle.joint_values if paired else oracle.losses if method=="minshap" else oracle.value
            ctx.native_batch=lambda masks:oracle.isolated_batch(observe,masks,pool)
            if method=="minshap":
                ctx.native_losses_batch=ctx.native_batch
        if paired:
            from minppi_mofs.baselines._permutation import select_permutation_pair
            answers=select_permutation_pair(ctx,joint_values=oracle.joint_values)
        elif method=="shapley":
            # Historical permutation experiments used this score; the formal baseline is SamplingSHAP.
            from minppi_mofs.baselines._permutation import select_permutation
            answers={method:select_permutation(ctx,method)}
        elif method=="full":
            answers={method:Selection(selected=columns,diagnostics=dict(baseline="full model"))}
        else:
            answers={method:get_selector(method)(ctx)}
        if pool is not None:
            progress("permutations_reduced_pool_closing")
    if native_workers>1:
        progress("pool_joined")
    estimate_s=time.perf_counter()-tick
    stats=oracle.describe()
    stats.update(native_workers=native_workers,peak_inflight_fits=ledger.peak_inflight,
                 native_executor="spawn_processes" if pool_pids else "serial",native_fit_pids=pool_pids)
    shared_fits=ledger.finished
    selection_fits=shared_fits-1
    proposals={}
    sizing_times={}
    for m in methods:
        tick=time.perf_counter()
        if meter is not None:
            meter.set_phase("sizing")
        if native_workers>1:
            progress("sizing_"+m)
        candidate,info=_candidate(answers[m],ctx,oracle)
        if meter is not None:
            stats.update(_prediction_usage(m, info, meter))
        info["seed"]=seed
        sizing_times[m]=time.perf_counter()-tick
        proposals[m]=candidate
        info["oracle"]=stats
        if paired:
            info["estimate_scope"]="shared_permutation_pair"
        atomic_write_json(folders[m]/"estimator.json",info)
    selector_s=time.perf_counter()-selector_begin
    if native_workers>1:
        progress("estimators_saved")
    del oracle,ctx,answers

    # Subsets shared by both methods require one guard fit and one final fit.
    # Each method records the physical fits it depended on. Only the joint total
    # may be summed with independent experiments.
    def group_supports(supports):
        groups={}
        for m in methods:
            groups.setdefault(tuple(supports[m]),[]).append(m)
        return groups

    dependent_fits={m:shared_fits for m in methods}
    results={}
    guards={}
    selected_by_method={}
    for names,users in group_supports(proposals).items():
        if paired:
            ledger.fit_users=users
        progress("guard",methods=users)
        tick=time.perf_counter()
        candidate=list(names)
        if candidate==columns:
            gp=full_guard
        elif not candidate:
            gp=_constant(yt,len(yg),regression)
        else:
            model=ledger.fit(policy,_matrix(train.loc[:,candidate]),yt,"guard")
            gp=_predict(model,_matrix(guard.loc[:,candidate]),regression)
            del model
            for m in users:
                dependent_fits[m]+=1
        candidate_scores=score(yg,gp)
        fail=_failures(ref,candidate_scores)
        fallback=enforce_guard and bool(fail)
        selected=columns if fallback else candidate
        chosen_guard=full_guard if fallback else gp
        selected_scores=ref if fallback else candidate_scores
        guard_s=time.perf_counter()-tick
        for m in users:
            frozen=dict(dataset=dataset,method=m,mode=mode,seed=seed,policy=policy,enforce_guard=enforce_guard,
                        validation_loss_limit=.05,fallback=fallback,candidate=candidate,selected=selected,
                        validation_full=ref,validation_candidate=candidate_scores,validation_selected=selected_scores,
                        failed_metrics=fail,rank_rows_sha256=canonical_hash(rank_rows.tolist()),
                        guard_rows_sha256=canonical_hash(guard_rows.tolist()))
            digest=canonical_hash(frozen)
            atomic_write_json(folders[m]/"frozen.json",frozen)
            guards[m]=(gp,chosen_guard)
            selected_by_method[m]=selected
            results[m]=dict(dataset=dataset,method=m,mode=mode,status="completed",seed=seed,width=len(columns),
                            candidate_k=len(candidate),candidate=candidate,k=len(selected),selected=selected,
                            selection_fits=selection_fits,reference_s=reference_s,estimate_s=estimate_s,
                            sizing_s=sizing_times[m],selector_s=selector_s,guard_s=guard_s,
                            enforce_guard=enforce_guard,validation_loss_limit=.05,fallback=fallback,
                            failed_metrics=fail,validation_full=ref,validation_candidate=candidate_scores,
                            validation_selected=selected_scores,frozen_hash=digest,**stats)

    # Every result's support is already signed on disk before either reads test labels.
    test=None
    y_final=np.concatenate([yt,yv])
    for names,users in group_supports(selected_by_method).items():
        if paired:
            ledger.fit_users=users
        progress("final",methods=users)
        tick=time.perf_counter()
        selected=list(names)
        if selected:
            x_final=np.concatenate([_matrix(frame if selected==columns else frame.loc[:,selected]) for frame in (train,val)])
            final=ledger.fit(policy,x_final,y_final,"final")
            del x_final
            for m in users:
                dependent_fits[m]+=1
        for m in users:
            atomic_write_json(folders[m]/"test_access.json",dict(frozen_hash=results[m]["frozen_hash"],
                              test_x_sha256=meta["sha256"]["test.npy"],test_y_sha256=meta["sha256"]["test_y.npy"]))
        if test is None:
            test,ye=load_split(source_root,dataset,"test")
            if list(test.columns)!=columns:
                raise ValueError("Held-out columns differ from training")
        if selected:
            test_pred=_predict(final,_matrix(test if selected==columns else test.loc[:,selected]),regression)
            del final
        else:
            test_pred=_constant(y_final,len(ye),regression)
        test_scores=score(ye,test_pred)
        final_s=time.perf_counter()-tick
        for m in users:
            gp,chosen_guard=guards[m]
            np.savez_compressed(folders[m]/"predictions.npz",rank_rows=rank_rows,guard_rows=guard_rows,
                                y_guard=yg,guard_full=full_guard,guard_candidate=gp,guard_selected=chosen_guard,
                                y_test=ye,test_pred=test_pred)
            n=dependent_fits[m]
            results[m].update(fits=n,fits_started=n,fit_count=n,final_s=final_s,test_scores=test_scores)
    elapsed=time.perf_counter()-started
    for m,res in results.items():
        res["elapsed_s"]=elapsed
        if paired:
            res.update(timing_scope="joint_active_time_both_methods",fit_scope="physical_fits_used_by_this_method",
                       joint_fits=ledger.finished,shared_reference_and_estimation_fits=shared_fits)
        atomic_write_json(folders[m]/"result.json",res)
    if paired:
        result=dict(dataset=dataset,mode=mode,method=method,status="completed",seed=seed,fits=ledger.finished,
                    fits_started=ledger.initiated,elapsed_s=time.perf_counter()-started,
                    timing_scope="one joint process; see watchdog.json for end-to-end cold time",
                    enforce_guard=enforce_guard,shared_reference_and_estimation_fits=shared_fits,
                    method_fit_counts=dependent_fits,methods={m:dict(k=r["k"],frozen_hash=r["frozen_hash"],
                        test_scores=r["test_scores"],directory=m) for m,r in results.items()},**stats)
        atomic_write_json(out/"result.json",result)
        progress("completed",methods=list(methods))
        return result
    progress("completed",k=results[method]["k"])
    return results[method]
