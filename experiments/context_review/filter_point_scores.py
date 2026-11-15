"""Recover exact standalone first-panel scores and check their already-sealed nominees."""
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import transport_predict as primary
from experiments.context_review.filter_neighborhood import ROOT, DEST
from experiments.context_review.forest_predict import new_forest, transform
from experiments.context_review.receiver_transport import SOURCE


def collect():
    lock = bg.read(DEST/"prediction-lock.json")
    rows = {}
    with threadpool_limits(limits=1):
        for key,rec in lock["folds"].items():
            source = ROOT/"results/context_review"/SOURCE[rec["panel"]]/key.split("/",1)[1]
            assert bg.digest(source/"inputs.npz")==rec["inputs_sha"]
            with np.load(source/"inputs.npz") as inputs:
                x,y,tr,proposal = (inputs[k] for k in ("x","y","train","proposal"))
            seed = 20421+rec["fold"]
            xt,xp = transform(x,tr,x[proposal])
            model = new_forest(seed).fit(xt,y[tr])
            ballot = primary.build_ballot(model,xt,xp,y[proposal],seed,rec["supports"][0],
                                          "ap" in rec["metrics"],point=True)
            scores = [float(primary.rating(ballot,s,[frozenset()],"point")) for s in rec["supports"][1:]]
            assert not scores or 1+np.argmax(scores)==rec["winners"]["point"]
            rows[key] = scores
    bg.store(DEST/"point-scores.json",dict(prediction_seal=bg.digest(DEST/"prediction-lock.json"),scores=rows))


if __name__=="__main__":
    collect()
