"""Check receipts, scores and support identities and collect PMLB evidence without feature tables."""
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from experiments.context_study.data import ROOT, OUT, digest


def add_study(archive, folder, expected_folds, arms):
    table = pd.read_csv(folder/"per-fold.csv")
    if len(table) != expected_folds*len(arms) or set(table.arm) != set(arms):
        raise ValueError(f"Incomplete method grid in {folder}")
    if table.duplicated(["dataset","fold","arm"]).any() or json.loads((folder/"failures.json").read_bytes()):
        raise ValueError("Duplicated evaluation or failure in released batch")
    for (name,fold), group in table.groupby(["dataset","fold"]):
        if set(group.arm)!=set(arms):
            raise ValueError("Missing method in fold")
        path = folder/name/f"fold-{fold}"
        completed = json.loads((path/"completed.json").read_bytes())
        selection = json.loads((path/"selection.json").read_bytes())
        if (completed["selection_sha256"]!=digest(path/"selection.json")
                or completed["predictions_sha256"]!=digest(path/"predictions.npz")
                or completed["protocol_sha256"]!=digest(folder/"protocol-lock.json")):
            raise ValueError(f"Invalid fold hashes in {path}")
        with np.load(path/"predictions.npz") as pred:
            y = pred["y"]
            np.testing.assert_array_equal(pred["row_id"],selection["test"])
            for row in group.itertuples():
                frozen = selection["selected"]
                support = frozen[row.arm] if isinstance(frozen,dict) else frozen
                if sorted(support)!=json.loads(row.selected) or len(support)!=row.k:
                    raise ValueError(f"Saved feature identities differ in {path}")
                prob=pred[row.arm]
                score=roc_auc_score(y,prob[:,1]) if prob.shape[1]==2 else roc_auc_score(y,prob,multi_class="ovr")
                np.testing.assert_allclose([score,accuracy_score(y,prob.argmax(axis=1))],[row.auc,row.accuracy],atol=1e-12,rtol=0.)
        for file in ["predictions.npz","completed.json","selection.json"]:
            archive.write(path/file,(path/file).relative_to(ROOT).as_posix())
    for file in ["protocol-lock.json","sources.json","per-fold.csv","duplicates.csv","unique-test-only.csv","failures.json"]:
        path=folder/file
        if path.exists():
            archive.write(path,path.relative_to(ROOT).as_posix())


def main():
    parent=OUT.parent
    configs=ROOT/"configs/context_study"
    args=json.loads((configs/"prospective.json").read_text(encoding="utf-8-sig"))
    grouped=json.loads((configs/"grouped.json").read_text(encoding="utf-8-sig"))
    target=ROOT/"packages/thesis-pmlb-evidence-r1.zip"
    tmp=target.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        add_study(archive,OUT,42,args["arms"])
        add_study(archive,parent/"prospective-pmlb-rfecv-r1",42,["rfecv"])
        add_study(archive,parent/"prospective-pmlb-duplicate-groups-r1",12,grouped["arms"])
        add_study(archive,parent/"prospective-pmlb-antichain-matched-r1",42,["antichain_matched_gain"])
        for name in ["prospective.json","duplicate-audit.md","grouped.json","rfecv.json","antichain-matched.json"]:
            path=configs/name
            archive.write(path,path.relative_to(ROOT).as_posix())
        source=ROOT/"experiments/context_study/README.md"
        archive.write(source,source.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(tmp) as archive:
        if archive.testzip() is not None:
            raise ValueError("Invalid evidence ZIP")
    tmp.replace(target)
    print(f"Verified 630 evaluations; {target.name}, SHA256 {digest(target)}")


if __name__=="__main__":
    main()

