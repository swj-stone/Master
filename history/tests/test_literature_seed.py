"""Guard against mixing seed-specific partitions, ranking draws or estimator policies."""
import hashlib
import json

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from experiments import literature_worker as worker


def test_custom_root_and_seed_propagate_to_split_estimators_and_receipts(tmp_path):
    for seed in (42,73):
        root=tmp_path/f"seed-{seed}"
        folder=root/"madelon"
        folder.mkdir(parents=True)
        rng=np.random.default_rng(seed)
        width=7
        parts={"train":620,"val":560,"test":80}
        meta=dict(dataset="madelon",seed=seed,width=width,features=[f"c{i}" for i in range(width)],
                  utility="roc_auc",partitions={k:dict(rows=n) for k,n in parts.items()},
                  policy=dict(seed=seed,backend="cpu",task="binary",model_params=dict(n_estimators=4,n_jobs=1)),
                  sha256={})
        for role,n in parts.items():
            x=rng.normal(size=(n,width)).astype("float32")
            y=(x[:,0]+x[:,2]>.1).astype("int64")
            for name,array in ((role,x),(role+"_y",y)):
                file=folder/f"{name}.npy"
                np.save(file,array)
                meta["sha256"][file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        (folder/"metadata.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError,match="does not match"):
            worker.trial("madelon","frozen","shapley",tmp_path/"invalid",seed=seed+1,input_root=root)
        out=tmp_path/f"output-{seed}"
        result=worker.trial("madelon","frozen","shapley",out,seed=seed,input_root=root)
        estimator=json.loads((out/"estimator.json").read_bytes())
        frozen=json.loads((out/"frozen.json").read_bytes())
        assert estimator["seed"]==frozen["seed"]==frozen["policy"]["seed"]==result["seed"]==seed
        digest=hashlib.sha256()
        schedule=np.random.default_rng(seed)
        for _ in range(4):
            digest.update(schedule.permutation(width).astype("<u4").tobytes())
        assert estimator["permutation_sha256"]==digest.hexdigest()
        yval=np.load(folder/"val_y.npy")
        rank,guard=train_test_split(np.arange(len(yval)),test_size=.5,random_state=seed,stratify=yval)
        with np.load(out/"predictions.npz") as pred:
            np.testing.assert_array_equal(pred["rank_rows"],np.sort(rank))
            np.testing.assert_array_equal(pred["guard_rows"],np.sort(guard))
            np.testing.assert_array_equal(pred["y_test"],np.load(folder/"test_y.npy"))
