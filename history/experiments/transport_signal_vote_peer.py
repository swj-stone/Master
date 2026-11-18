"""Exploratory transfer check of a train-only dominant-peer replication screen."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import signal_vote_benchmark as benchmark
from experiments import signal_vote_peer_mechanism as simulation
from experiments.adaptive_minpi_transfer import Panel
from experiments.stable_minpi_stress import ROOT, file_hash, write_json
from minppi_mofs.signal_vote_minpi import signal_vote_top_from_training


DIRECTORIES = dict(original="signal-vote-69-20261117a", transfer="signal-vote-transfer-20261117a",
                   mechanism="signal-vote-peer-timing-20261117a")
ARMS = ("signal_auto","F_signal","no_peer","random_signal")
KEY = ["dataset","fold","seed"]


def correlated_peers(X, strongest):
    """Center observed fit rows, replacing missing entries by their fit-row column mean."""
    x = np.asarray(X,dtype=float)
    finite = np.isfinite(x)
    count = finite.sum(axis=0)
    anchor = x[finite.argmax(axis=0), np.arange(x.shape[1])]
    anchor = np.where(count,anchor,0.)
    centered = np.subtract(x,anchor,out=np.zeros_like(x),where=finite)
    shift = centered.sum(axis=0)/np.maximum(count,1)
    np.subtract(centered,shift,out=centered,where=finite)
    norm = np.sqrt(np.einsum("ij,ij->j",centered,centered))
    products = np.abs(centered.T@centered[:,strongest])
    match = (norm>0) & (norm[strongest]>0) & (products >= .90*norm*norm[strongest])
    match[strongest] = False
    return int(match.sum())


def panels(suite):
    if suite != "mechanism":
        yield from benchmark.panels(suite)
        return
    for task in simulation.TASKS:
        for q in simulation.PROXIES:
            for fold,seed in enumerate(simulation.SEEDS,1):
                X,y,Xt,truth,_ = simulation.generated(task,q,seed)
                yield Panel(f"{task}-q{q}","synthetic","proxy",task,fold,simulation.SELECTOR_SEED,
                            X,y,Xt,truth,{})


def inspect(suite, folder):
    rows = []
    for panel in panels(suite):
        n,p = panel.X.shape
        ordered = panel.name in ("gas","secom") if suite == "original" else panel.ordered
        cols,info = signal_vote_top_from_training(panel.X,panel.y,task=panel.task,
                                                  random_state=panel.seed,groups=panel.groups,ordered_groups=ordered)
        with np.load(folder/f"{panel.name}-{panel.fold}-{panel.seed}.npz") as saved:
            np.testing.assert_array_equal(cols,saved["signal_auto_cols"])
        counts, anchors = [], []
        for direction in info["folds"]:
            peer = direction["peers"][0]
            anchors.append(int(peer))
            counts.append(correlated_peers(panel.X[direction["fit_rows"]],peer))
        minimum = min(counts,default=0)
        qualifies = p >= 32 and n >= 512 and bool(counts) and 3*minimum >= p
        rows.append(dict(suite=suite,**benchmark.case_id(panel),task=panel.task,n_train=n,p=p,
                         width=len(cols),strongest_columns=json.dumps(anchors),other_peer_counts=json.dumps(counts),
                         directions=len(counts),minimum_share=minimum/p,meets_size=p>=32 and n>=512,
                         qualifies=qualifies))
    return pd.DataFrame(rows)


def compare(membership, folder):
    cases = pd.read_csv(folder/"cases.csv")
    cases = cases[cases.method.isin(ARMS)]
    frame = cases.merge(membership,on=KEY,validate="many_to_one",suffixes=("","_stored"))
    np.testing.assert_array_equal(frame.width,frame.width_stored)
    np.testing.assert_array_equal(frame.p,frame.p_stored)
    draws = pd.read_csv(folder/"random-draws.csv")
    if "random_score" not in draws:
        draws = draws.rename(columns={"score":"random_score"})
    paired = draws[KEY+["draw","random_score"]].merge(frame[KEY+["method","score"]],on=KEY)
    paired["win"] = paired.score > paired.random_score+1e-12
    comparison = paired.groupby(KEY+["method"],as_index=False).agg(draws=("draw","size"),wins=("win","sum"))
    result = frame.merge(comparison,on=KEY+["method"],validate="one_to_one")
    result["strict"] = result.joint & (4*result.wins >= 3*result.draws)
    return result


def main(out,verify):
    folders = {k: ROOT/"results/minpi-study"/v for k,v in DIRECTORIES.items()}
    frozen = [Path(__file__),ROOT/"src/minppi_mofs/signal_vote_minpi.py", Path(benchmark.__file__),
              Path(simulation.__file__)]
    frozen += [f/name for f in folders.values() for name in ("cases.csv","protocol.json","random-draws.csv")]
    state = dict(exploratory="Defined after observing q=48 peer-mechanism results; transport and calibration, "
                             "not a prospective or independently confirmed population.",
                 rule="At least 512 training rows and 32 columns. In every existing voting direction, "
                      "use only that direction's fitting rows and fitting-selected strongest column. "
                      "Require at least one third of p columns, excluding self, to have |Pearson r| >= .90 "
                      "with that column. Impute by the fitting rows' mean; constant columns do not match. "
                      "Take minimum share across directions. The downstream time is the previously "
                      "measured pipeline; these descriptive subgroup counts do not charge a routing diagnostic.",
                 strict="Original joint criterion and score exceeds 75% or more of the frozen same-width "
                        "random draws by > 1e-12 (20 per case on original/transfer, 32 on mechanism).",
                 files={p.relative_to(ROOT).as_posix():file_hash(p) for p in frozen})
    if verify:
        assert json.loads((out/"protocol.json").read_text())==state
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/"protocol.json",state)
    memberships, comparisons = [], []
    for suite,folder in folders.items():
        membership = inspect(suite,folder)
        assert len(membership)=={"original":69,"transfer":57,"mechanism":20}[suite]
        memberships.append(membership)
        comparisons.append(compare(membership,folder))
    membership=pd.concat(memberships,ignore_index=True)
    frame=pd.concat(comparisons,ignore_index=True)
    tables={"membership.csv":membership,"method-cases.csv":frame}
    for keys,name in ((["suite","qualifies","method"],"summary.csv"),
                      (["suite","dataset","qualifies","method"],"by-source.csv")):
        tables[name]=frame.groupby(keys,as_index=False).agg(configs=("fold","size"),sources=("dataset","nunique"),
            reduced=("reduced","sum"),within=("within","sum"),joint=("joint","sum"),strict=("strict","sum"),
            random_wins=("wins","sum"),draws=("draws","sum"),median_ratio=("ratio","median"))
    for name,t in tables.items():
        if verify:
            pd.testing.assert_frame_equal(t,pd.read_csv(out/name),check_dtype=False)
        else:
            t.to_csv(out/name,index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(1):
        main(args.output,args.verify)
