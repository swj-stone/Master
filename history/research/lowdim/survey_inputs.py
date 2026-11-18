"""Adapters for the original surveys, advertisement and formal simulation archives."""
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import yaml

from experiments import minpi_constant_transfer as advertising
from experiments import signal_vote_confirmation as surveys
from experiments.minpi_study import OUTPUT, P, ROOT, file_hash, load_dataset, train
from experiments.stable_minpi_stress import digest
from minppi_mofs.experiment_protocol import canonical_hash
from research.lowdim.formal_study import inputs


@dataclass
class StudyCase:
    dataset: str
    source: str
    family: str
    fold: int
    seed: int
    X: np.ndarray
    y: np.ndarray
    Xt: np.ndarray
    truth: np.ndarray
    task: str
    metric: str
    expected_full: np.ndarray
    traces: dict
    impute: bool = False


def study_cases():
    for dataset in ("sparse_signal", "xor", "null"):
        raw, target, task, _, colmap, _, _ = load_dataset(dataset)
        for fold, seed in enumerate(P["formal_seeds"], 1):
            X, y, Xt, truth, tr, te, pred, decision = inputs(dataset, seed, raw, target, colmap, task)
            yield StudyCase(dataset, dataset, "formal_simulation", fold, seed, X, y, Xt, truth,
                            task, "auc", pred, dict(train=tr, test=te, archived_decision=decision))

    for module, task, label, idx in ((surveys.anes96, "classification", "vote", 0),
                                      (surveys.fair, "regression", "affairs", 1)):
        for p in surveys.survey_panels(module, task, label, idx):
            path = OUTPUT/"signal-vote-confirmation-20261117a"/f"{p.name}-{p.fold}-{p.seed}.npz"
            with np.load(path) as old:
                hashes = [digest(a) for a in (p.X, p.y, p.Xt, p.truth)]
                np.testing.assert_array_equal(hashes, old["input_hashes"])
                np.testing.assert_array_equal(p.truth, old["truth"])
                pred = old["full_pred"]
            yield StudyCase(p.name, p.name.split("_")[0], p.family, p.fold, p.seed, p.X, p.y,
                            p.Xt, p.truth, p.task, "balanced_accuracy" if idx == 0 else "r2", pred,
                            dict(original_cols=p.split_info["source_cols"],
                                 train_digest=p.split_info["train_digest"], test_digest=p.split_info["test_digest"]))

    for name, fold, seed, X, y, Xt, truth, task, split in advertising.panels():
        if name.startswith("generated"):
            break
        if name.startswith("internet-advertisements"):
            path = OUTPUT/"constant-transfer-20261117a"/f"{name}-{fold}.npz"
            with np.load(path) as old:
                for key, value in dict(**split, truth=truth).items():
                    np.testing.assert_array_equal(value, old[key])
                pred = old["full_pred"]
            yield StudyCase(name, "internet-advertisements", "nested_train", fold, seed, X, y, Xt,
                            truth, task, "balanced_accuracy", pred, split, impute=True)


def fraud_cases():
    conf = yaml.safe_load((ROOT/"configs/credit_card_fraud.yaml").read_bytes())["dataset"]
    raw_path = ROOT.parent/conf["raw_path"]
    assert raw_path.stat().st_size == conf["raw_byte_count"] and file_hash(raw_path)==conf["raw_sha256"]
    folder = ROOT.parent/"results/icomp2026_formal/icomp2026-formal-v1-s42-123/preprocess/credit_card_fraud/s42"
    split=json.loads((folder/"split_metadata.json").read_bytes())
    assert split["seed"]==42 and split["dataset_id"]=="credit_card_fraud"
    xy=pd.read_csv(raw_path)
    cols=split["selected_features"]
    target=split["target"]
    labels=xy[target].to_numpy(dtype=np.int64)
    data=xy[cols].to_numpy(dtype=np.float64)
    del xy
    rows={}
    for part in ("train","val","test"):
        idx=split[part+"_indices"]
        assert canonical_hash(idx)==split[part+"_index_sha256"]
        rows[part]=np.array(idx,dtype=np.int64)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(rows.values()))),np.arange(len(labels)))
    prepared={}
    for part in ("train","test"):
        idx=rows[part]
        frame=pd.read_csv(folder/(part+".csv"))
        assert list(frame.columns)==[*cols,target]
        np.testing.assert_array_equal(frame.pop(target).to_numpy(),labels[idx])
        prepared[part]=np.ascontiguousarray(frame.to_numpy())
        scaled=(data[idx]-split["scaler_mean"])/split["scaler_scale"]
        np.testing.assert_allclose(prepared[part],scaled,atol=1e-11,rtol=1e-11)
    tr,te=rows["train"],rows["test"]
    # An independent rerun on the frozen, pre-scaled CSV verifies the Full labels.
    # The conference fit used a different learner and cannot supply Ridge predictions.
    pred,_=train("ridge",prepared["train"],labels[tr],prepared["test"],"classification",42)
    yield StudyCase("credit_card_fraud","credit_card_fraud","conference_extension",1,42,
                    data[tr],labels[tr],data[te],labels[te],"classification","auc",pred,
                    dict(train=tr,test=te,feature_names=cols,raw_sha=conf["raw_sha256"]))
