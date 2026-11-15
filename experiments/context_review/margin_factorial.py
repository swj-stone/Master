"""Complete the two-by-two comparison of panels and restorations on frozen candidate slates."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import transport_predict as policy
from experiments.context_review.ballot_refinement import settings

EVIDENCE=bg.ROOT/"thesis-paper/evidence/margin_factorial"
DESIGN=bg.ROOT/"configs/context_review/margin_factorial.json"


def nominate(panel):
    src,_=settings(panel)
    plan_dir="transport" if panel=="first" else "wide"
    receipt=bg.ROOT/"thesis-paper/evidence"/plan_dir/"prediction-lock.json"
    keys=sorted(p.parent.relative_to(src).as_posix() for p in src.glob("*/fold-*/selection.json"))
    assert receipt.is_file() and len(keys)==(30 if panel=="first" else 16)
    records=[]
    for key in keys:
        f=src/key
        state=bg.read(f/"selection.json")
        layout=bg.read(f/"layout.json")
        assert state["layout"]==bg.digest(f/"layout.json") and layout["inputs_sha256"]==bg.digest(f/"inputs.npz")
        choice=state["choices"]["gain"]
        anchor=tuple(choice["anchor"])
        winners={cell:anchor for cell in ("empty1","empty4","family1","family4")}
        if state["pool"]:
            with np.load(f/"inputs.npz") as data:
                x,y,train,prop=[data[c] for c in ("x","y","train","proposal")]
            imp=SimpleImputer(strategy="median",keep_empty_features=True).fit(x[train])
            xt,xp=imp.transform(x[train]),imp.transform(x[prop])
            seed=20421+layout["fold"]
            model=bg.fit(xt,y[train],seed)
            ballot=policy.build_ballot(model,xt,xp,y[prop],seed,anchor,choice["rare"])
            family=policy.separating_contexts(choice["order"][len(anchor):])
            best={cell:-np.inf for cell in winners}
            for row in state["pool"]:
                cols=frozenset(row["columns"])
                values=[]
                for restoration in family:
                    a,b=ballot.score(cols|restoration),ballot.score(ballot.anchor|restoration)
                    margins=[policy.panel_margin(av,bv) for av,bv in zip(a,b,strict=True)]
                    values.append((margins[0],sorted(margins)[1]))
                    if not restoration:
                        empty=values[-1]
                scores=dict(empty1=empty[0],empty4=empty[1],family1=min(v[0] for v in values),
                            family4=min(v[1] for v in values))
                np.testing.assert_allclose([scores["empty1"],scores["family4"]],[row["point"],row["slack"]],atol=1e-10,rtol=1e-12)
                for cell,score in scores.items():
                    if score>best[cell]:
                        best[cell],winners[cell]=score,tuple(row["columns"])
        assert list(winners["empty1"])==state["choices"]["point"]["nominee"]
        assert list(winners["family4"])==state["choices"]["slack"]["nominee"]
        records.append(dict(key=key,selection_hash=bg.digest(f/"selection.json"),winners=winners))
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    bg.store(EVIDENCE/(panel+"-lock.json"),dict(design=bg.digest(DESIGN),code=bg.digest(Path(__file__)),
            original=bg.digest(receipt),folds=records))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel",choices=("first","wide"))
    with threadpool_limits(limits=1):
        nominate(parser.parse_args().panel)
