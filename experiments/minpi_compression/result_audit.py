"""Independently verify sealed supports, refitted predictions and equal-source contrasts."""
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from experiments.minpi_compression.sources import DATA, ROOT, load


EVIDENCE = ROOT / "thesis-paper/evidence/minpi_compression"
DESIGN = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf8"))


def outcomes(y, pred, task, scale):
    if task == "classification":
        auc = []
        for i in range(pred.shape[1]):
            pos = y == i
            n, m = pos.sum(), (~pos).sum()
            auc.append((rankdata(pred[:, i])[pos].sum() - n*(n+1)/2) / (n*m))
        return dict(auc=np.mean(auc), accuracy=np.mean(y == np.argmax(pred, axis=1)))
    sse = np.square(y - pred).sum()
    mae = np.abs(y - pred).mean()
    return dict(r2=1.-sse/np.square(y-y.mean()).sum(), mae=mae, scaled_mae=mae/scale)


def earliest_width(inner, p, task, strict=False):
    widths = sorted(map(int, inner))
    assert widths == sorted({1, *(int(np.ceil(p*f)) for f in DESIGN["width_fractions"])})
    full = inner[str(p)]
    cls_gap, reg_gap = (0.,0.) if strict else (.01,.02)
    for k in widths:
        s = inner[str(k)]
        if task == "classification":
            passed = s["auc"] >= full["auc"]-cls_gap and s["accuracy"] >= full["accuracy"]-cls_gap
        else:
            passed = s["r2"] >= full["r2"]-reg_gap and s["scaled_mae"] <= full["scaled_mae"]+reg_gap
        if passed:
            return k
    raise ValueError("Full support missing or invalid")


def verify(bundle):
    published = pd.read_csv(EVIDENCE / "confirmation/scores.csv").set_index(["dataset","seed","method"])
    strict = pd.read_csv(EVIDENCE / "zero_tolerance/confirmation/scores.csv").set_index(["dataset","seed","method"])
    lock = json.loads((EVIDENCE / "confirmation/lock.json").read_bytes())
    assert lock == json.loads(bundle.read("lock.json"))
    rows, witnesses, strict_rows = [], [], []
    for name in DESIGN["confirmation"]:
        assert sha256((DATA/f"{name}.zip").read_bytes()).hexdigest() == DESIGN["source_sha256"][name]
        x, y, task, groups = load(name)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        for seed in DESIGN["seeds"]:
            key = f"{name}-{seed}"
            receipt = bundle.read(f"{key}/selection.json")
            assert sha256(receipt).hexdigest() == lock[key]
            rec = json.loads(receipt)
            payload = bundle.read(f"{key}/predictions.npz")
            assert sha256(payload).hexdigest() == rec["predictions"]
            assert rec["seed"] == seed and rec["dataset"] == name and task == rec["task"]
            split = {k:np.array(v) for k,v in rec["splits"].items()}
            np.testing.assert_array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(x)))
            assigned = [set(groups[v]) for v in split.values()]
            assert all(not (a & b) for i,a in enumerate(assigned) for b in assigned[i+1:])
            dev = np.sort(np.r_[split["fit"], split["vote"], split["width"]])
            test = split["test"]
            trainx, testx = x[dev], x[test]
            scale = trainx.std(axis=0)
            scale[scale < 1e-12] = 1.
            z = (trainx-trainx.mean(axis=0))/scale
            v = (testx-trainx.mean(axis=0))/scale
            t = np.eye(len(np.unique(y[dev])))[y[dev]] if task == "classification" else y[dev]
            with np.load(BytesIO(payload)) as predictions:
                for method in DESIGN["methods"]:
                    runs = [row for row in rec["rows"] if row["method"] == method]
                    assert len(runs) == DESIGN["repetitions"]
                    row = runs[0]
                    k, order = row["k"], row["order"]
                    assert all(r["k"] == k and r["order"] == order and r["inner"] == row["inner"] for r in runs)
                    assert sorted(order) == list(range(x.shape[1]))
                    if method != "full":
                        assert k == earliest_width(row["inner"], len(order), task)
                    cols = sorted(order[:k])
                    pred = predictions[method]
                    independent = Ridge(alpha=1.,solver="svd").fit(z[:, cols], t).predict(v[:, cols])
                    np.testing.assert_allclose(pred, independent, rtol=1e-9, atol=1e-9)
                    metrics = outcomes(y[test], pred, task, y[dev].std())
                    clocks = {key:np.median([r[key] for r in runs])
                              for key in ("rank_s","selection_s","deployment_s")}
                    record = dict(dataset=name, seed=seed, method=method, task=task,
                                  k=k, p=len(order), retained=k/len(order), **metrics, **clocks)
                    score = published.loc[name,seed,method]
                    assert list(map(int, str(score["selected"]).split())) == cols
                    for key in [*metrics,*clocks,"k","p","retained"]:
                        assert np.isclose(record[key],score[key],rtol=1e-12,atol=1e-13), (name,seed,method,key)
                    rows.append(record)
                    tight = len(order) if method == "full" else earliest_width(row["inner"],len(order),task,True)
                    scols = sorted(order[:tight])
                    selected = strict.loc[name,seed,method]
                    assert selected["k"] == tight >= k
                    assert list(map(int,str(selected["selected"]).split())) == scols
                    if tight != k:
                        spred = Ridge(alpha=1.,solver="svd").fit(z[:, scols], t).predict(v[:, scols])
                    else:
                        spred = independent
                    smetrics = outcomes(y[test], spred, task, y[dev].std())
                    for metric in smetrics:
                        assert np.isclose(smetrics[metric],selected[metric],rtol=1e-9,atol=1e-9), (name,seed,method,metric)
                    strict_rows.append(dict(dataset=name,seed=seed,method=method,task=task,k=tight,**smetrics))
            ranks = {r["method"]:r for r in rec["rows"] if r["repeat"] == 0}
            m, c = ranks["minpi"], ranks["coefficient"]
            mcols, ccols = set(m["order"][:m["k"]]), set(c["order"][:c["k"]])
            work = m["work"]
            assert work["scheduled_marginals"] == sum(work[key] for key in
                ("absorbed_skips", "losing_base_skips", "evaluated_marginals"))
            witnesses.append(dict(dataset=name, seed=seed, task=task, p=len(m["order"]), k=m["k"],
                                  negative=m["minimum"].count(-1), zero=m["minimum"].count(0),
                                  positive=m["minimum"].count(1), reordered=m["order"] != c["order"],
                                  changed_support=mcols != ccols,
                                  retained_negative=sum(m["minimum"][j]==-1 for j in mcols),
                                  **{key:work[key] for key in ("scheduled_marginals","absorbed_skips",
                                      "losing_base_skips","evaluated_marginals")}))
    assert len(lock) == len(witnesses) and len(rows) == len(published)
    return pd.DataFrame(rows), pd.DataFrame(witnesses), pd.DataFrame(strict_rows)


def compare(frame):
    source_rows, macro_rows = [], []
    for task, domain in frame.groupby("task"):
        outcomes_ = ("auc","accuracy") if task == "classification" else ("r2","scaled_mae")
        proposed = domain[domain.method=="minpi"].set_index(["dataset","seed"])
        for baseline in sorted(set(domain.method)-{"minpi"}):
            other = domain[domain.method==baseline].set_index(["dataset","seed"])
            associations = {k:proposed[k]-other[k] for k in outcomes_}
            associations["width_ratio"] = np.log(proposed.k/other.k)
            # The full model selects all columns without computing a ranking.
            if baseline != "full" and "selection_s" in domain:
                associations["selection_time_ratio"] = np.log(proposed.selection_s/other.selection_s)
            if "deployment_s" in domain:
                associations["deployment_time_ratio"] = np.log(proposed.deployment_s/other.deployment_s)
            source = pd.DataFrame(associations).groupby("dataset").mean()
            for metric in source:
                conversion = np.exp if metric.endswith("_ratio") else np.asarray
                vals = source[metric].values
                redraws = [np.mean(draw) for draw in product(vals,repeat=len(vals))]
                lo, hi = conversion(np.quantile(redraws,[.025,.975]))
                macro_rows.append(dict(task=task,baseline=baseline,measure=metric,
                                       estimate=conversion(vals.mean()).item(),lo=lo,hi=hi))
                for name,val in source[metric].items():
                    source_rows.append(dict(task=task,dataset=name,baseline=baseline,measure=metric,
                                            estimate=conversion(val).item()))
    return pd.DataFrame(source_rows),pd.DataFrame(macro_rows)


def main():
    with threadpool_limits(limits=1), ZipFile(EVIDENCE/"confirmation/forecasts.zip") as bundle:
        frame, witnesses, strict = verify(bundle)
    source, macro = compare(frame)
    _, strict_macro = compare(strict)
    out = EVIDENCE / "audit"
    out.mkdir(exist_ok=True,parents=True)
    source.to_csv(out/"source-effects.csv",index=False)
    macro.to_csv(out/"paired-effects.csv",index=False)
    strict_macro.to_csv(out/"strict-paired-effects.csv",index=False)
    witnesses.to_csv(out/"vote-decisions.csv",index=False)
    print(macro[macro.baseline.isin(["full","coefficient","rfe"])].to_string(index=False))
    print(witnesses.groupby("task").sum(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
