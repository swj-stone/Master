"""Verify every prospective receipt and repeated forecast before releasing held scores."""
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score

from experiments.minpi_replication.second_panel import data


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/anchored_vote"
OUT = ROOT / "thesis-paper/evidence/anchored_vote"
METHODS = ("min", "mean", "gain", "gain_grid", "exhaustive")
SOURCES = ("vowel", "hcv", "shoppers", "garment", "tetouan")
SEEDS = tuple(range(42, 47))
REPEATS = range(3)


def digest(raw):
    return sha256(raw).hexdigest()


def validate_support(cols, width):
    if not cols or cols != sorted(set(cols)) or min(cols) < 0 or max(cols) >= width:
        raise ValueError(f"Invalid {width}-column support {cols}")


def require(condition, description):
    if not condition:
        raise ValueError(description)


def load_panel():
    require(set(data.SOURCES) == set(SOURCES) and tuple(data.SEEDS) == SEEDS, "Unregistered panel")
    panel, ledger = {}, []
    signatures = set()
    for repeat in REPEATS:
        root = RUN/f"prospective-r{repeat}"
        expected = {root/f"{n}-{s}"/m/"receipt.json" for n,s,m in product(SOURCES, SEEDS, METHODS)}
        observed = set(root.glob("*/*/receipt.json"))
        require(expected == observed, f"{root}: {len(expected-observed)} missing, {len(observed-expected)} extra")
    for name, seed in product(SOURCES, SEEDS):
        x, blind, task, idx = data.split(name, seed)
        test = np.asarray(idx["test"])
        ids = np.asarray(data.row_ids(name))[test]
        require(len(np.unique(ids)) == len(ids), f"Repeated original row IDs for {name}/{seed}")
        require(np.all(blind[test] == -1) if task == "classification" else np.isnan(blind[test]).all(),
                f"Missing outer-label mask for {name}/{seed}")
        classes = np.unique(blind[idx["fit"]]).size
        shape = (len(test), classes) if task == "classification" else (len(test),)
        for method in METHODS:
            for rep in REPEATS:
                directory = RUN/f"prospective-r{rep}"/f"{name}-{seed}"/method
                raw = (directory/"receipt.json").read_bytes()
                rec = json.loads(raw)
                sig = json.dumps(rec["signature"],sort_keys=True)
                signatures.add(sig)
                require((name,seed,method,task,x.shape[1],"cpu") ==
                        tuple(rec[key] for key in ("dataset","seed","method","task","p","device")),
                        f"Inconsistent identity at {directory}")
                archive = directory/"forecast.npz"
                require(digest(archive.read_bytes()) == rec["forecast"], f"Corrupted forecasts at {directory}")
                r = rec["record"]
                validate_support(r["selected"], x.shape[1])
                require(r["k"] == len(r["selected"]) and np.isfinite(r["utility"]), f"Bad decision at {directory}")
                require(0 < r["selection_s"] <= r["seconds"] and np.isfinite(r["seconds"]),
                        f"Bad runtime at {directory}")
                if method in {"min","mean","gain"}:
                    validate_support(r["anchor"],x.shape[1])
                    curve = r["anchor_curve"]
                    floor = max(curve.values())-.005
                    anchor_k = min(int(k) for k,v in curve.items() if v >= floor-1e-14)
                    require(abs(r["floor"]-floor) < 1e-12 and len(r["anchor"]) == anchor_k
                            and set(r["selected"]).issubset(r["anchor"]) and r["utility"] >= floor-1e-14
                            and r["fits"] <= min(x.shape[1],6)+1, f"Failed anchor guarantee at {directory}")
                forecasts = {}
                with np.load(archive,allow_pickle=False) as arr:
                    np.testing.assert_array_equal(arr["indices"], test)
                    needed = {"primary","near"} if method == "exhaustive" else {"primary"}
                    require(set(arr.files) == needed|{"indices"}, f"Unexpected fields at {directory}")
                    for field in needed:
                        preds = np.asarray(arr[field])
                        require(preds.shape == shape and np.isfinite(preds).all(),f"Invalid prediction at {directory}")
                        if task == "classification":
                            require(np.all((preds >= 0)&(preds <= 1)),f"Non-probability at {directory}")
                            np.testing.assert_allclose(preds.sum(axis=1), 1., rtol=0., atol=3e-6)
                        forecasts[field]=preds
                if method == "exhaustive":
                    validate_support(r["near_support"],x.shape[1])
                    curve=r["trace"]
                    require(set(map(int,curve)) == set(range(1,x.shape[1]+1)),f"Incomplete Gain path {directory}")
                    for slack,key in ((0,"selected"),(.005,"near_support")):
                        expected_k = min(int(k) for k,u in curve.items() if u >= max(curve.values())-slack-1e-14)
                        require(len(r[key])==expected_k,f"Wrong exhaustive width {directory}/{key}")
                    require(r["fits"] == x.shape[1]+1 and r["near_seconds"] >= r["selection_s"]
                            and np.isfinite(r["near_seconds"]),f"Bad Gain cost {directory}")
                key = name,seed,method
                if rep == 0:
                    panel[key] = (r,forecasts,task)
                else:
                    first,ref,_ = panel[key]
                    for field in ("selected","utility","trace","fits","anchor","floor","anchor_curve","near_support"):
                        require(first.get(field) == r.get(field),f"Repetition changed {field}: {directory}")
                    for field,preds in forecasts.items():
                        np.testing.assert_array_equal(ref[field],preds)
                ledger.append(dict(dataset=name,seed=seed,method=method,repeat=rep,seconds=r["seconds"],
                                   near_seconds=r.get("near_seconds"),selection_seconds=r["selection_s"],
                                   receipt_sha256=digest(raw),forecast_sha256=rec["forecast"],
                                   original_rows_sha256=digest(json.dumps(ids.tolist()).encode()),
                                   indices_sha256=digest(np.asarray(test,dtype="<i8").tobytes()),
                                   signature_sha256=digest(sig.encode())))
        base=panel[name,seed,"gain"][0]
        fullcurve=panel[name,seed,"exhaustive"][0]["trace"]
        for m in ("gain","min","mean"):
            arm=panel[name,seed,m][0]
            for field in ("anchor","anchor_curve","floor"):
                require(arm[field]==base[field],f"Unmatched {field} at {name}/{seed}")
            for k,value in arm["anchor_curve"].items():
                require(value==fullcurve[k],f"Mismatch in native Gain utility {name}/{seed}/{k}")
    require(len(signatures)==1, "Code signature changed during the three timed repetitions")
    return panel,pd.DataFrame(ledger)


def score(truth,preds,task):
    if task == "regression":
        return dict(primary=r2_score(truth,preds),r2=r2_score(truth,preds),mae=mean_absolute_error(truth,preds))
    target_classes=np.unique(truth)
    np.testing.assert_array_equal(target_classes,np.arange(preds.shape[1]))
    auc=roc_auc_score(truth,preds[:,1] if len(target_classes)==2 else preds,multi_class="ovr",average="macro")
    return dict(primary=auc,auc=auc,accuracy=accuracy_score(truth,preds.argmax(axis=1)))


def release(panel,ledger):
    rows=[]
    for name,seed in product(SOURCES, SEEDS):
        truth=np.asarray(data.truth(name,seed))
        outputs={}
        for method in METHODS:
            r,forecasts,task=panel[name,seed,method]
            times=ledger[(ledger.dataset==name)&(ledger.seed==seed)&(ledger.method==method)]
            for field,pred in forecasts.items():
                near=field=="near"
                support=tuple(r["near_support"] if near else r["selected"])
                if support in outputs:
                    np.testing.assert_array_equal(pred,outputs[support])
                outputs[support]=pred
                rows.append(dict(dataset=name,seed=seed,method="exhaustive_near" if near else method,
                                 task=task,k=len(support),fits=r["fits"],anchor_k=len(r.get("anchor",[])) or None,
                                 columns=json.dumps(support,separators=(",",":")),rows=len(truth),
                                 predictions_sha256=digest(np.ascontiguousarray(pred).tobytes()),
                                 seconds=times["near_seconds" if near else "seconds"].median(),
                                 **score(truth,pred,task)))
    return pd.DataFrame(rows)


def contrasts(rows):
    primary = rows[rows.method=="min"]
    results=[]
    for method in (*METHODS[1:],"exhaustive_near"):
        joined=primary.merge(rows[rows.method==method],on=["dataset","seed"],suffixes=("_min","_control"),
                             validate="one_to_one")
        require(len(joined) == len(SOURCES)*len(SEEDS), "Missing matched seed")
        joined["delta_seconds"]=joined.seconds_min-joined.seconds_control
        joined["width_excess"]=joined.k_min-joined.k_control-np.maximum(1.,.05*joined.k_control)
        joined["delta_primary"]=joined.primary_min-joined.primary_control
        speed=joined.delta_seconds<0
        width=joined.width_excess<=0
        qual=joined.delta_primary>=-.01
        for name,g in joined.groupby("dataset"):
            ix=g.index
            passes = g.delta_seconds.median()<0 and g.width_excess.median()<=0
            result=dict(dataset=name,control=method,task=g.task_min.iloc[0],source_pass=passes,
                        source_pass_quality=passes and g.delta_primary.median()>=-.01,
                        source_pass_5pct=bool((g.seconds_min/g.seconds_control).median()<=.95
                                              and g.width_excess.median()<=0 and g.delta_primary.median()>=-.01),
                        simultaneous_seeds=int((speed[ix]&width[ix]).sum()),
                        simultaneous_quality_seeds=int((speed[ix]&width[ix]&qual[ix]).sum()),
                        anchor_reductions=int((g.k_min<g.anchor_k_min).sum()),
                        changed_supports=int((g.columns_min!=g.columns_control).sum()),
                        changed_forecasts=int((g.predictions_sha256_min!=g.predictions_sha256_control).sum()),
                        median_paired_time_ratio=(g.seconds_min/g.seconds_control).median(),
                        mean_source_pass=(g.delta_seconds.mean()<0 and
                                          g.k_min.mean()-g.k_control.mean()<=max(1,.05*g.k_control.mean())))
            for c in ("seconds_min","seconds_control","k_min","k_control","fits_min","fits_control","delta_seconds",
                      "width_excess","delta_primary","primary_min","primary_control","auc_min","auc_control",
                      "accuracy_min","accuracy_control","r2_min","r2_control","mae_min","mae_control"):
                result[c]=g[c].median() if g[c].notna().any() else np.nan
            results.append(result)
    return pd.DataFrame(results)


def main():
    panel,ledger=load_panel()
    rows=release(panel,ledger)
    sources=contrasts(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in (("receipts",ledger),("seeds",rows),("sources",sources)):
        frame.to_csv(OUT/f"{name}.csv",index=False)
    print(sources.groupby("control")[["source_pass","source_pass_quality","simultaneous_seeds",
                                      "simultaneous_quality_seeds","changed_supports","changed_forecasts"]].sum())


if __name__ == "__main__":
    main()
