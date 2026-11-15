"""Seal or verify the live timing sample and compare it to historical mask counts."""
import argparse
from hashlib import sha256
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost

from experiments.context_review import budgeted_guard as bg
from experiments.context_review import filter_neighborhood, forest_predict, matched_stopping, neighborhood_replicates


DEST = matched_stopping.DEST


def text_digest(path):
    return sha256(Path(path).read_text(encoding="utf8").encode("utf8")).hexdigest()


def environment():
    processor = platform.processor()
    if platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            processor = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return dict(cpu=processor, operating_system=platform.platform(), architecture=platform.machine(),
                python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                forest_n_jobs=1, threadpoolctl_limit=1, cpu_affinity="not pinned")


def audit(record=False):
    receipt_path = DEST/"timing-receipt.json"
    folds = pd.read_csv(DEST/"folds.csv")
    first = pd.read_csv(filter_neighborhood.DEST/"costs.csv")
    gain = pd.read_csv(forest_predict.DEST/"costs.csv").assign(design="gain")
    original = pd.concat([gain,first[first.delivery=="guarded"].assign(design="anova")])
    original = original[original.method.isin(["vote","slack","point"])].copy()
    original["pruned"] = original.method.eq("vote")
    original["method"] = original.method.replace({"slack":"margin"})
    keys = ["panel","dataset","fold","design","method","pruned"]
    old = original.merge(folds[folds.draw==1],on=keys,validate="one_to_one")
    assert len(old)==276
    np.testing.assert_array_equal(old.masks_x, old.masks_y)
    times = pd.read_csv(DEST/"timings.csv")
    assert (times.nomination_seconds>=0).all() and len(times)==3*len(folds)==3864
    group = ["panel","dataset","fold","design","draw","repeat"]
    sizes = times.groupby(group).position.agg(["size","nunique","min","max"]).drop_duplicates()
    assert sizes.values.tolist() == [[7,7,1,7]]
    lock = bg.read(forest_predict.DEST/"prediction-lock.json")
    fold_order = {name:i for i,name in enumerate(lock["folds"])}
    for (panel,dataset,fold,design,draw,repeat), block in times.groupby(group):
        index = fold_order[f"{panel}/{dataset}/fold-{fold}"]
        slate_index = 0 if design=="gain" else draw
        rng = np.random.default_rng(np.random.SeedSequence([37189,index,slate_index,repeat-1]))
        expected = [matched_stopping.ARMS[i] for i in rng.permutation(7)]
        order = block.sort_values("position")
        assert list(zip(order.method,order.pruned,strict=True)) == expected
    matched_stopping.summarize_times(folds)
    files = [matched_stopping.DESIGN, Path(matched_stopping.__file__),
             *(p/"prediction-lock.json" for p in (forest_predict.DEST,filter_neighborhood.DEST,neighborhood_replicates.DEST))]
    values = dict(timing_csv_utf8_lf_sha256=text_digest(DEST/"timings.csv"),
                  source_and_inputs_utf8_lf_sha256={f.relative_to(bg.ROOT).as_posix():text_digest(f) for f in files},
                  slates=184, historical_costs_reproduced=len(old), repetitions=3,
                  live_nomination_runs=len(times), distinct_input_folds=46)
    if record:
        bg.store(receipt_path,{**values,"timing_csv_original_sha256":bg.digest(DEST/"timings.csv"),
                               "environment":environment()})
    else:
        stored = bg.read(receipt_path)
        assert values == {key:stored[key] for key in values}
    print("Verified 3,864 live choices and counts, all 552 prespecified method permutations, and all 276 historical costs.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record",action="store_true",help="Record a receipt after a completed local timing run")
    audit(p.parse_args().record)
