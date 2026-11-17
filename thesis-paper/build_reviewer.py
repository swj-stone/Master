"""Assemble and verify the separate, whitelisted score-only reviewer supplement."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tree_release as release


PAPER,ROOT = release.PAPER,release.ROOT
TARGET = release.PACKAGES/"minpi-reviewer-scores.zip"


def evaluation_labels():
    from experiments.minpi_rework import field_cost, validation_predictive, wearable_data, wide_veto_data

    aligned = {}
    def keep(panel, key, ids, y):
        prefix = panel+"/"+key
        aligned[prefix+"/rows"] = ids.astype("<i8")
        aligned[prefix+"/y"] = y[ids]
    for panel,src,fn in (("seven",validation_predictive,validation_predictive._partition),
                         ("wide",wide_veto_data,wide_veto_data.partition)):
        for name in src.SOURCES:
            for seed in range(42,47):
                _,y,_,idx = fn(name,seed)
                keep(panel,f"{name}-{seed}",idx["test"],y)
    for width in (12,13):
        for seed in range(42,47):
            _,y,idx,*_ = field_cost.source(width,seed,hide=False)
            keep("medical",f"p{width}-{seed}",idx["test"],y)
    for name,(_,_,n,_) in wearable_data.SOURCES.items():
        _,truth,subjects,_ = wearable_data.read(name)
        for fold in range(n):
            key = f"{name}-{fold}"
            rec = json.loads((ROOT/"results/minpi_rework/wearable-groups"/key/"selection.json").read_bytes())
            ids = np.flatnonzero(np.isin(subjects,rec["people"]["test"]))
            keep("wearable",key,ids,truth)
    return aligned


def published():
    evidence = release.EVIDENCE
    seven = pd.read_csv(evidence/"predictive-confirmation/scores.csv").rename(columns={"method":"arm"})
    seven = seven[seven.arm.isin(["full","mdi_min","mdi_plus_min_min"])].copy()
    wide = pd.read_csv(evidence/"predictive-reachable/scores.csv")
    wide = wide[(wide.panel=="wide")&wide.arm.isin(["full","mdi_original","portfolio_min_original"])].copy()
    wide = wide.rename(columns={"reachable":"k"})
    wide["arm"] = wide.arm.replace({"mdi_original":"mdi","portfolio_min_original":"portfolio"})
    for df in (seven,wide):
        df["job"] = df.dataset+"-"+df.seed.astype(str)
    clinical = pd.read_csv(evidence/"field-cost/scores.csv").rename(columns={"width":"k"})
    clinical["job"] = "p"+clinical.p.astype(str)+"-"+clinical.seed.astype(str)
    wear = pd.read_csv(evidence/"wearable-groups/held.csv")
    wear["job"] = wear.dataset+"-"+wear.fold.astype(str)
    bounded = pd.read_csv(evidence/"wearable-groups/refit-held.csv").rename(
        columns={"method":"arm","acc":"accuracy","held_worst":"worst_risk"})
    bounded["job"] = bounded.dataset+"-"+bounded.fold.astype(str)
    cols = ["panel","job","arm","k","auc","accuracy","r2","mae","fee","worst_risk"]
    return pd.concat([df.assign(panel=name).reindex(columns=cols) for name,df in
                     (("seven",seven),("wide",wide),("medical",clinical),("wearable",wear),("wearable_refit",bounded))],
                     ignore_index=True).sort_values(cols[:3])


def inspect_zip(z):
    """Inspect paths, ZIP metadata, text and array headers, recursively."""
    local = re.compile(rb"(?:\b[A-Za-z]:(?:[/\\])|/(?:home|Users)/|\.SPBU|ZhaoCHI|MastersThesis|"
                       rb"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b)",re.I)
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    def scan(data, location):
        if local.search(data) or (len(user)>=4 and user.lower().encode() in data.lower()):
            raise ValueError(f"Found local path, mail address or local account information in {location}")
    scan(z.comment,"ZIP comment")
    for item in z.infolist():
        name = item.filename
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unexpected ZIP member path {name}")
        scan(name.encode()+item.comment+item.extra,name)
        raw = z.read(name)
        if name.endswith((".zip",".npz")):
            with ZipFile(BytesIO(raw)) as inner:
                inspect_zip(inner)
        elif name.endswith(".npy"):
            scan(raw[:128],name)
        elif name.endswith((".json",".md",".txt",".py",".csv")):
            scan(raw,name)


def build():
    with TemporaryDirectory(dir=ROOT/"tmp",prefix="reviewer-build-") as temp:
        work = Path(temp)
        np.savez_compressed(work/"held-labels.npz",**evaluation_labels())
        published().to_csv(work/"published.csv",index=False)
        files = {name:work/name for name in ("held-labels.npz","published.csv")}
        files.update({"README.md":PAPER/"reviewer-scores.md","replay.py":PAPER/"reviewer_scores.py",
                      "requirements.txt":PAPER/"reviewer-requirements.txt",
                      "seven.zip":release.PACKAGES/"minpi-predictive-seven-forecasts.zip",
                      "wide.zip":release.PACKAGES/"minpi-wide-veto-forecasts.zip",
                      "reachable.zip":release.PACKAGES/"minpi-reachable-forecasts.zip",
                      "field-cost.zip":release.EVIDENCE/"field-cost/forecast-receipts.zip",
                      "wearable.zip":release.PACKAGES/"minpi-wearable-forecasts.zip",
                      "wearable-refit.zip":release.PACKAGES/"minpi-wearable-refit-forecasts.zip"})
        for name,folder,pin in (("seven","predictive-confirmation","forecast-sha256.txt"),
                                ("wide","wide-veto","archive-sha256.txt"),
                                ("reachable","predictive-reachable","archive-sha256.txt")):
            assert release.digest(files[name+".zip"])==(release.EVIDENCE/folder/pin).read_text().split()[0]
        assert release.digest(files["field-cost.zip"])==json.loads((release.EVIDENCE/"field-cost/lock.json").read_bytes())["archive"]
        with ZipFile(files["wearable.zip"]) as z:
            for src in ("mhealth","dsads"):
                key = f"lock-{src}.json"
                assert z.read(key)==(release.EVIDENCE/"wearable-groups"/key).read_bytes()
        assert release.digest(files["wearable-refit.zip"])==(release.EVIDENCE/"wearable-groups/refit-archive-sha256.txt").read_text().split()[0]
        release.archive(TARGET,files)


def verify():
    with TemporaryDirectory(dir=ROOT/"tmp",prefix="reviewer-scores-") as temp, ZipFile(TARGET) as z:
        inspect_zip(z)
        work = Path(temp)
        z.extractall(work)
        # -I removes the project, user site and inherited PYTHONPATH from imports.
        subprocess.run([sys.executable,"-I","replay.py"],cwd=work,check=True)
        print("Checked isolated replay, ZIP entry names, nested text and array headers.",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("build","verify"))
    (build if parser.parse_args().action=="build" else verify)()
