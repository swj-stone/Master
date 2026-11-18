"""Time the original and current reserve baselines with identical inputs and work budgets."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.run_thesis_sampling import ROOT
from experiments.test.piv3_raw_splits import _sha, load_split
from minppi_mofs.experiment_protocol import atomic_write_json


ORIGINAL_REVISION = "dbe06e9517eb7c85664fc1c0e7aa810a7c66934e"


def _original_module(relative, revision):
    """Import a pinned Git blob without switching worktrees or changing the shared environment."""
    if revision != ORIGINAL_REVISION:
        raise ValueError(f"Expected the pre-audit baseline revision {ORIGINAL_REVISION}")
    source = subprocess.check_output(["git", "show", f"{revision}:src/minppi_mofs/{relative}.py"], cwd=ROOT)
    name = "_equal_work_original_" + relative.replace("/", "_")
    module = ModuleType(name)
    module.__file__ = f"{revision}:{relative}.py"
    sys.modules[name] = module  # dataclasses resolves class metadata through its defining module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def measure(inputs, dataset, method, out, revision=ORIGINAL_REVISION):
    from importlib.metadata import version
    from minppi_mofs import modeling
    from minppi_mofs.baselines import nsga2_fs, shap_top_k

    if method not in ("tree_before", "tree_after", "nsga_before", "nsga_after"):
        raise ValueError(f"Unknown equal-work run {method}")
    tree = method.startswith("tree_")
    old = method.endswith("_before")
    module = shap_top_k if tree else nsga2_fs
    if old:
        model_api, model_hash = _original_module("modeling", revision)
        module, module_hash = _original_module("baselines/" + ("shap_top_k" if tree else "nsga2_fs"), revision)
        if tree:
            module.make_model = model_api.make_model
        else:
            module.evaluate_fixed_subset = model_api.evaluate_fixed_subset
    else:
        model_hash, module_hash = _sha(Path(modeling.__file__)), _sha(Path(module.__file__))
    inputs, out = Path(inputs), Path(out)
    source = inputs/"inputs/seed-42"/dataset
    meta = json.loads((source/"metadata.json").read_bytes())
    pin = json.loads((inputs/"protocol.json").read_bytes())["inputs"]["42"][dataset]
    if (meta["dataset"] != dataset or meta["seed"] != 42 or meta["input_type"] != "archived_preprocessed_csv"
            or _sha(source/"metadata.json") != pin["metadata_sha256"]):
        raise ValueError(f"Data partition does not match protocol: {source}")
    for part in (("train",) if tree else ("train","val")):
        for file in (part+".npy", part+"_y.npy"):
            if _sha(source/file) != meta["sha256"][file]:
                raise ValueError(f"Input hash differs: {source/file}")
    train, yt = load_split(source.parent, dataset, "train")
    policy = meta["policy"]
    work = dict(dataset=dataset, seed=42, method=method, width=train.shape[1], train_rows=len(train),
                revision=revision if old else "working tree", metadata_sha256=pin["metadata_sha256"],
                selector_sha256=module_hash, modeling_sha256=model_hash, xgboost=version("xgboost"),
                backend=policy["backend"])

    def progress(phase, **fields):
        atomic_write_json(out/"progress.json", dict(phase=phase, **fields))

    if tree:
        diag = {}
        progress("fit_and_explain_all_training_rows")
        start = time.perf_counter()
        scores = module.compute_shap_topk_importance(train, yt, **policy, diagnostics=diag,
                                                     **({} if old else {"max_rows":None}))
        work.update(selector_s=time.perf_counter()-start, fits=diag["model_fit_count"], scores=scores,
                    explained_rows=len(train))
    else:
        val, yv = load_split(source.parent, dataset, "val")
        rank, _ = train_test_split(np.arange(len(val)), test_size=.5, random_state=42,
                                   stratify=None if policy["task"] == "regression" else yv)
        rank = np.sort(rank)
        ranked, yr = val.iloc[rank], yv[rank]
        fit_count = 0

        def evaluated(*, logical_evaluation_count, unique_subset_count, memoization_hit_count, selected_count, cached):
            nonlocal fit_count
            fit_count += int(not cached and selected_count > 0)
            progress("search", done=logical_evaluation_count, total=24, fits=fit_count,
                     unique=unique_subset_count, cache_hits=memoization_hit_count)

        if old:
            # The original search calls its evaluator only on cache misses.
            evaluate = module.evaluate_fixed_subset
            def tracked(*args, **kwargs):
                nonlocal fit_count
                score = evaluate(*args, **kwargs)
                fit_count += int(bool(args[4]))
                progress("search", fits=fit_count, total=24)
                return score
            module.evaluate_fixed_subset = tracked
        progress("search", done=0, total=24, fits=0)
        start = time.perf_counter()
        front = module.nsga2_feature_selection(train, yt, ranked, yr, **policy, pop_size=8, max_gen=2,
                                               **({} if old else {"progress":evaluated}))
        work.update(selector_s=time.perf_counter()-start, fits=fit_count, rank_rows=len(rank), population=8,
                    generations=2, unique=front.unique_subset_count, requests=front.logical_evaluation_count,
                    cache_hits=front.memoization_hit_count, pareto_size=len(front.pareto_front), hv=front.hv,
                    hv_history=list(front.hv_history),
                    front=[(mask.tolist(), list(obj)) for mask,obj in front.pareto_front])
    atomic_write_json(out/"result.json", work)
    progress("completed", fits=work["fits"])
    return work
