"""Post-hoc sensitivity using age and the three lesion attributes, omitting clinician BI-RADS."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.context_review import transport_predict as policy
from experiments.context_review.transport_data import EVIDENCE, ROOT, RUN, read
from experiments.context_review.transport_score import fold_rows
from experiments.context_study.benchmark import store
from experiments.context_study.data import digest

NEW = ROOT / "results/context_review/transport-mammographic-four"
NAME = "mammographic"
DESCRIPTION = ("Unplanned sensitivity after reading the primary results: omit first input "
               "(radiologist's BI-RADS assessment), retain age, shape, margin and density. "
               "Identical original row IDs and class encodings; recompute learner and anchor, "
               "four half-width proposals, guards and final fits on this four-column input. "
               "For aggregate impact, replace only Mammographic's three folds among the 30.")


def predict():
    NEW.mkdir(parents=True, exist_ok=True)
    lock = dict(description=DESCRIPTION, code=digest(Path(__file__)),
                primary_seal=digest(EVIDENCE / "prediction-lock.json"))
    store(NEW / "protocol.json", lock)
    protocol = digest(NEW / "protocol.json")
    policy.RUN = NEW
    receipt = {}
    for fold in range(3):
        original = RUN / NAME / f"fold-{fold}"
        folder = NEW / NAME / f"fold-{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        meta = read(original / "layout.json")
        if not (folder / "selection.json").exists():
            with np.load(original / "inputs.npz") as data:
                prepared = {key:data[key] for key in data.files}
            x, xo = prepared["x"][:,1:], prepared["x_outer"][:,1:]
            assert x.shape[1] == xo.shape[1] == 4
            prepared.update(x=x, x_outer=xo)
            np.savez_compressed(folder / "inputs.npz", **prepared)
            full = pd.util.hash_pandas_object(pd.DataFrame(np.r_[x,xo]), index=False).to_numpy()
            meta.update(p=4, missing=int(np.isnan(x).sum()+np.isnan(xo).sum()),
                        duplicated_inputs=int(len(full)-len(np.unique(full))),
                        overlap_test_rows=int(np.isin(full[len(x):],full[:len(x)]).sum()),
                        inputs_sha256=digest(folder / "inputs.npz"))
            store(folder / "layout.json", meta)
        policy.predict_fold(NAME, fold, 1161, (12+fold)%4, None, protocol)
        receipt[str(fold)] = digest(folder / "selection.json")
    store(NEW / "prediction-lock.json", dict(protocol=protocol, selections=receipt))
    print("All three four-input prediction archives saved.")


def score():
    lock = read(NEW / "prediction-lock.json")
    records = []
    for fold in range(3):
        original = RUN / NAME / f"fold-{fold}"
        folder = NEW / NAME / f"fold-{fold}"
        meta = read(folder / "layout.json")
        assert digest(folder / "selection.json") == lock["selections"][str(fold)]
        assert digest(folder / "predictions.npz") == read(folder / "selection.json")["predictions"]
        assert digest(original / "targets.npz") == meta["targets_sha256"]
        # Targets enter only after the new 3-fold prediction seal has been written.
        (folder / "targets.npz").write_bytes((original / "targets.npz").read_bytes())
        rows, _, _ = fold_rows(folder)
        records.extend(rows)
    raw = pd.DataFrame(records)
    raw.to_csv(EVIDENCE / "birads-four-features.csv", index=False)
    primary = pd.read_csv(EVIDENCE / "fold-results.csv")
    replacement = pd.concat([primary[primary.dataset != NAME], raw], ignore_index=True)
    fields = ["u", "breach","auc","accuracy","eligible","passed","k","retained"]
    means = replacement.groupby(["dataset","arm"])[fields].mean().groupby(level="arm").mean()
    means.to_csv(EVIDENCE / "birads-replaced-summary.csv")
    store(EVIDENCE / "birads-sensitivity-lock.json", dict(description=DESCRIPTION, **lock,
          predictions={str(fold):digest(NEW / NAME / f"fold-{fold}/predictions.npz") for fold in range(3)}))
    print(raw.groupby("arm")[fields].mean().to_string())
    print(means.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["predict", "score"])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (predict if args.phase == "predict" else score)()
