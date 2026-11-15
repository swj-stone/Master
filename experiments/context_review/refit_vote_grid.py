"""Vote on paired refitted-model predictions over frozen compact-Gain prefixes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review.refit_target import metrics
from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import DATA, OUT as PRIMARY, ROOT, digest
from minppi_mofs.piv3_compact_witness import _within


DESIGN = ROOT/"configs/context_review/refit_vote_grid.json"
DONOR = ROOT/"results/context_review/refit-guard"
OUT = ROOT/"results/context_review/refit-vote-grid"
ARMS = ("global_point", "class_point", "class_vote")


def read(path):
    return json.loads(path.read_bytes())


def panels(y, seed):
    classes = [np.flatnonzero(y==c) for c in range(int(y.max())+1)]
    if not all(len(group) for group in classes):
        raise ValueError("A class is absent in the guard")
    rng = np.random.default_rng(np.random.SeedSequence([seed,659]))
    return [np.arange(len(y))]+[np.concatenate([rng.choice(group,len(group),replace=True)
                                               for group in classes]) for _ in range(3)]


def measure(y, probabilities, rare):
    predicted = probabilities.argmax(axis=1)
    recall = np.bincount(y,predicted==y,minlength=probabilities.shape[1])/np.bincount(y,minlength=probabilities.shape[1])
    if not np.isfinite(recall).all():
        raise ValueError("Missing class in a reported evaluation")
    return metrics(y,probabilities,rare),recall


def check_votes(votes, required):
    """Visit a context's panels until it passes or can no longer pass."""
    checked = 0
    for context in votes:
        yes = 0
        for j,passed in enumerate(context,1):
            checked += 1
            yes += int(passed)
            if yes>=required:
                break
            if yes+len(context)-j<required:
                return False,checked
    return True,checked


def select(votes_by_width, arm, anchor_k):
    trace = []
    actual = anchor_k
    for width,table in sorted(votes_by_width.items()):
        if width==anchor_k:
            break
        votes = table if arm=="class_vote" else table[:,:1]
        if arm=="global_point":
            votes = votes[:1]
        required = 3 if arm=="class_vote" else 1
        passed,checked = check_votes(votes,required)
        assert passed == (votes.sum(axis=1).min()>=required)
        trace.append(dict(k=width,passed=passed,checks=checked,possible=int(votes.size)))
        if passed:
            actual = width
            break
    return actual,trace


def fit_prefixes(x, raw, class_index, train, evaluation, supports, seed):
    started = perf_counter()
    transformer = SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
    xt,xe = transformer.transform(x[train]),transformer.transform(x[evaluation])
    y = raw.iloc[train].map(class_index).to_numpy()
    preparation_s = perf_counter()-started
    predictions, fit_s = {},{}
    for width,columns in sorted(supports.items()):
        started = perf_counter()
        predictions[width] = fit(xt[:,columns],y,seed).predict_proba(xe[:,columns])
        fit_s[width] = perf_counter()-started
    return predictions,fit_s,preparation_s


def run(name, fold):
    started = perf_counter()
    folder = OUT/name/f"fold-{fold}"
    folder.mkdir(parents=True,exist_ok=True)
    archive = DONOR/name/f"fold-{fold}"
    donor = read(archive/"proposals.json")
    assert digest(archive/"proposals.json")==read(archive/"selection.json")["proposals_sha256"]
    primary = PRIMARY/name/f"fold-{fold}"
    assert digest(primary/"selection.json")==donor["donor"]
    source = read(primary/"selection.json")
    data = DATA/name/f"{name}.tsv.gz"
    assert digest(data)==read(primary/"completed.json")["data_sha256"]
    table = pd.read_csv(data,sep="\t")
    raw = table.pop("target")
    x = table.to_numpy(dtype=np.float32)
    class_index = {c:i for i,c in enumerate(source["class_order"])}
    seed = 20421+fold
    train, proposal, guard, test = (np.asarray(donor[field]) for field in ("train","proposal","guard","test"))
    assert len(np.unique(np.r_[train,proposal,guard,test]))==len(x)==sum(map(len,(train,proposal,guard,test)))
    assert set(np.r_[proposal,guard])==set(source["rank"]) and set(test)==set(source["test"])
    k = len(donor["candidates"]["anchor"])
    widths = sorted({(j*k+3)//4 for j in (1,2,3,4)})
    supports = {w:sorted(donor["order"][:w]) for w in widths}
    assert supports[k]==donor["candidates"]["anchor"]
    inner = np.sort(np.r_[train,proposal])
    pg,fit_s,prep = fit_prefixes(x,raw,class_index,inner,guard,supports,seed)
    t = perf_counter()
    y_guard = raw.iloc[guard].map(class_index).to_numpy()
    samples = panels(y_guard,seed)
    values = {w:[measure(y_guard[ix],pred[ix],donor["rare"]) for ix in samples] for w,pred in pg.items()}
    votes = {}
    for w in widths:
        answers = []
        for (reference,original),(candidate,reduced) in zip(values[k],values[w],strict=True):
            answers.append([_within(candidate,reference),*(reduced+1e-12>=.97*original)])
        votes[w] = np.array(answers).T
    ballot_s = perf_counter()-t
    selected = {"anchor":k}
    traces = {}
    for arm in ARMS:
        selected[arm],traces[arm] = select(votes,arm,k)
    np.savez_compressed(folder/"guard.npz",rows=guard,y=y_guard,panels=np.array(samples),
                        **{f"k{w}":v for w,v in pg.items()})
    store(folder/"selection.json",dict(donor=digest(archive/"proposals.json"),rare=donor["rare"],
          protocol=digest(OUT/"protocol.json"),guard=digest(folder/"guard.npz"),supports=supports,
          votes={w:mat.tolist() for w,mat in votes.items()},widths=selected,traces=traces))
    frozen = digest(folder/"selection.json")
    # Only frozen supports can reach the outer evaluation.
    final_widths = read(folder/"selection.json")["widths"]
    p,outer_s,outer_prep = fit_prefixes(x,raw,class_index,np.sort(np.r_[inner,guard]),test,
                                        {w:supports[w] for w in set(final_widths.values())},seed)
    y_test = raw.iloc[test].map(class_index).to_numpy()
    scores = {w:measure(y_test,prob,donor["rare"]) for w,prob in p.items()}
    reference,r_ref = scores[k]
    rows = []
    for arm,w in final_widths.items():
        current,recall = scores[w]
        trace = traces.get(arm,[])
        rows.append(dict(dataset=name,fold=fold,arm=arm,k=w,anchor_k=k,width=x.shape[1],retained=w/x.shape[1],
                         exceeds=not _within(current,reference),class_exceeds=bool(np.any(recall+1e-12<.97*r_ref)),
                         recall=float(recall.mean()),checks=sum(m["checks"] for m in trace),
                         possible_checks=sum(m["possible"] for m in trace),**current))
    np.savez_compressed(folder/"predictions.npz",rows=test,y=y_test,**{f"k{w}":prob for w,prob in p.items()})
    assert digest(folder/"selection.json")==frozen
    store(folder/"completed.json",dict(rows=rows,selection=frozen,predictions=digest(folder/"predictions.npz"),
          guard_models=len(pg),final_models=len(p),guard_fit_predict_seconds=fit_s,
          development_fit_predict_seconds=outer_s,guard_prep_s=prep,outer_prep_s=outer_prep,
          complete_ballot_seconds=ballot_s,worker_seconds=perf_counter()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=2)
    args = parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):
            run(args.worker[0],int(args.worker[1]))
        return
    OUT.mkdir(exist_ok=True,parents=True)
    eligible = pd.read_csv(DONOR/"eligibility.csv")
    assert len(eligible)==42 and eligible.eligible.sum()==41
    sources = [Path(__file__),DESIGN,DONOR/"protocol.json",DONOR/"eligibility.csv",PRIMARY/"protocol-lock.json",
               ROOT/"experiments/context_review/refit_target.py",ROOT/"experiments/context_study/benchmark.py",
               ROOT/"src/minppi_mofs/piv3_compact_witness.py"]
    sources.extend(DONOR/r.dataset/f"fold-{r.fold}"/"proposals.json" for r in eligible.itertuples() if r.eligible)
    lock = {p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    protocol = OUT/"protocol.json"
    if protocol.exists() and read(protocol)!=lock:
        raise ValueError("Frozen inputs changed")
    store(protocol,lock)
    eligible.to_csv(OUT/"eligibility.csv",index=False)
    rows,costs = [],[]
    started = perf_counter()
    for index,r in enumerate(eligible.itertuples(),1):
        if not r.eligible:
            print(f"[{index}/42] {r.dataset} {r.fold} excluded: {r.reason}",flush=True)
            continue
        folder = OUT/r.dataset/f"fold-{r.fold}"
        if not (folder/"completed.json").exists():
            subprocess.run([sys.executable,"-m","experiments.context_review.refit_vote_grid","--worker",
                            r.dataset,str(r.fold)],check=True,timeout=3600)
        rec = read(folder/"completed.json")
        selection = read(folder/"selection.json")
        assert selection["protocol"]==digest(protocol) and selection["guard"]==digest(folder/"guard.npz")
        assert rec["selection"]==digest(folder/"selection.json") and rec["predictions"]==digest(folder/"predictions.npz")
        rows.extend(rec["rows"])
        costs.append(dict(dataset=r.dataset,fold=r.fold,inner_fits=rec["guard_models"],outer_fits=rec["final_models"],
                          inner_fit_s=sum(rec["guard_fit_predict_seconds"].values()),
                          outer_fit_s=sum(rec["development_fit_predict_seconds"].values()),
                          ballots_s=rec["complete_ballot_seconds"],wall_s=rec["worker_seconds"]))
        elapsed = perf_counter()-started
        print(f"[{index}/42] {r.dataset} {r.fold}, {rec['worker_seconds']:.1f}s, elapsed {elapsed:.0f}s"
              f", remaining estimate {elapsed*(42-index)/index:.0f}s, widths {selection['widths']}",flush=True)
    df = pd.DataFrame(rows)
    assert len(df)==41*4 and not df.duplicated(["dataset","fold","arm"]).any()
    df.to_csv(OUT/"per-fold.csv",index=False)
    fields = ["auc","accuracy","recall","retained","exceeds","class_exceeds","checks","possible_checks"]
    means = df.groupby(["dataset","arm"])[fields].mean()
    means.to_csv(OUT/"by-dataset.csv")
    means.groupby("arm").mean().to_csv(OUT/"summary.csv")
    complete = eligible.groupby("dataset").eligible.all()
    means.loc[means.index.get_level_values("dataset").isin(complete[complete].index)].groupby("arm").mean().to_csv(
        OUT/"complete-tasks.csv")
    pd.DataFrame(costs).to_csv(OUT/"costs.csv",index=False)
    print(df.groupby("arm")[["exceeds","class_exceeds","checks","possible_checks"]].sum().to_string(),flush=True)


if __name__=="__main__":
    main()
