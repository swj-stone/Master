"""Develop by validation, lock one global recipe, then cold-test PIv1 and its ablation."""
import argparse
import csv
import hashlib
import json
import multiprocessing
import shutil
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.run_piv0_benchmark import exclusive_run
from experiments.run_thesis_sampling import MSK, ROOT, supervise, write_summary
from experiments.test.run_piv0_seed42 import rank_model
from experiments.thesis_prepared_catalog import audit_prepared_catalog
from experiments.thesis_sampling_worker import subset_validation_scores, validation_halves
from minppi_mofs.baselines.shap_top_k import fitted_tree_shap_importance
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json, canonical_hash,
)
from minppi_mofs.modeling import fit_predict
from minppi_mofs.utility import compute_utility


DATASETS = ("madelon", "spambase", "abalone", "gisette", "credit_card_fraud", "isolet")
VOTE_RECIPES = ("vote", "vote_strong")
RECIPES = (*VOTE_RECIPES, "anchor")
DIAGNOSTICS = ("mean_vote", "mean_vote_strong")
FINAL_SEEDS = (42, 123)
LIMIT = 900
TIE = 1e-14
OUTPUT = ROOT / "results/test/thesis_sampling"
BASELINE = ROOT / "results/thesis_benchmark/PIv0-all-seeds-20261111-213840-398718"


def read(path):
    return json.loads(path.read_bytes())


def source_files():
    from minppi_mofs import piv1, piv1_search
    return [Path(__file__), Path(piv1.__file__), Path(piv1_search.__file__)]


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}


def choose_recipe(development):
    """Equal weight per dataset. The weaker vote wins aggregate ties."""
    totals = Counter({recipe: 0. for recipe in VOTE_RECIPES})
    for choices in development:
        left, right = (choices[name]["validation"][0] for name in VOTE_RECIPES)
        if abs(left-right) <= TIE:
            for name in VOTE_RECIPES:
                totals[name] += 1.5
        else:
            totals["vote"] += 1 if left > right else 2
            totals["vote_strong"] += 1 if right > left else 2
    return min(VOTE_RECIPES, key=totals.get), dict(totals)


def run_arm(source, row, policy, recipes, phase, out, expected):
    from minppi_mofs.piv1 import relative_votes
    from minppi_mofs.piv1_search import search_subsets

    seed, dataset, metric = int(row["seed"]), row["dataset_id"], row["utility"]
    stages = {}
    fits = 0

    def progress(name, **state):
        atomic_write_json(out/"progress.json", dict(phase=name, fits=fits, **state))

    progress("loading")
    train, yt, val, yv = load_prepared_train_validation(source, row)
    names = list(train.columns)
    width = len(names)
    full = tuple(range(width))
    halves, half_id = validation_halves(yv, policy["task"], seed=seed)
    progress("full_model")
    start = time.monotonic()
    model, safe_val, full_scores = rank_model(train, yt, val, yv, halves, metric, policy)
    fits += 1
    stages["full_model_s"] = time.monotonic()-start
    start = time.monotonic()
    progress("training_attributions")
    gain_map = model.get_booster().get_score(importance_type="gain")
    gain = np.array([gain_map.get(f"f{i}", 0.) for i in full])
    shap = fitted_tree_shap_importance(model, train.set_axis(safe_val.columns, axis=1), seed=seed, max_rows=1024)
    stages["attribution_s"] = time.monotonic()-start

    votes = None
    if any(name != "anchor" for name in recipes):
        start = time.monotonic()
        progress("voting")
        votes = relative_votes(model, safe_val, yv, metric=metric, seed=seed,
                               progress=lambda name,step,n: progress(name, checked=step, total=n))
        stages["voting_s"] = time.monotonic()-start
    else:
        stages["voting_s"] = 0.

    partition_sha = {part: row[f"{part}_file_sha256"] for part in ("train", "val", "test")}
    ranking = dict(dataset=dataset, seed=seed, metric=metric, source=source.name, policy=policy,
                   partition_sha256=partition_sha, halves=half_id, features=names,
                   gain=gain.tolist(), training_mean_abs_shap=shap.tolist(), votes=votes)
    atomic_write_json(out/"ranking.json", ranking)
    scores = {full: tuple(map(float,full_scores))}
    winners = {}
    start = time.monotonic()
    for nrecipe, recipe in enumerate(recipes):
        candidates_used = 0
        # The same retained mask is trained at most once across development recipes.
        def evaluate(indices):
            nonlocal fits,candidates_used
            key = tuple(map(int,indices))
            if not key or key != tuple(sorted(set(key))) or key[0] < 0 or key[-1] >= width:
                raise ValueError("Candidate masks must contain sorted, unique original indices")
            candidates_used += 1
            budget = min(24, max(6, int(np.ceil(2.4*np.sqrt(width)))))
            projected = (len(recipes)-nrecipe-1)*budget + max(0,budget-candidates_used)
            calls = nrecipe*budget+candidates_used
            progress("subset_validation", recipe=recipe, checked=candidates_used, total=budget,
                     eta_s=(time.monotonic()-start)*projected/calls)
            if key not in scores:
                columns = [names[i] for i in key]
                s = subset_validation_scores(train, yt, val, yv, columns, halves, utility=metric, **policy)
                scores[key] = tuple(map(float,s))
                if not np.isfinite(scores[key]).all():
                    raise ValueError("Validation utility is not finite")
                fits += 1
            return scores[key][0]

        search_recipe = recipe.removeprefix("mean_")
        score_key = "mean_vote" if recipe in DIAGNOSTICS else "min_vote"
        vote_scores = None if recipe == "anchor" else votes[score_key]
        consistent = None if recipe == "anchor" else votes["negative_consistent"]
        output = search_subsets(vote_scores, shap, gain, evaluate, recipe=search_recipe, consistent_negative=consistent)
        selected = tuple(map(int,output["selected"]))
        if abs(output["utility"]-scores[selected][0]) > TIE:
            raise ValueError("Selected subset has inconsistent utility")
        winners[recipe] = dict(indices=list(selected), selected=[names[i] for i in selected],
                               retained=len(selected), validation=scores[selected],
                               budget=output["budget"], base_budget=output["base_budget"], checked=len(output["trials"]),
                               source=output["winner_source"])
        atomic_write_json(out/f"trials-{recipe}.json", output["trials"])
    stages["search_s"] = time.monotonic()-start
    atomic_write_json(out/"validation.json", [
        dict(indices=list(key), scores=value) for key,value in scores.items()
    ])
    frozen = dict(dataset=dataset, seed=seed, metric=metric, width=width, full_validation=list(full_scores),
                  ranking_hash=canonical_hash(ranking), choices=winners, partition_sha256=partition_sha)
    identity = canonical_hash(frozen)
    atomic_write_json(out/"frozen.json", frozen)
    test_score = None
    if phase == "final":
        recipe, = recipes
        best = winners[recipe]
        if expected is not None and (list(expected["selected"]) != best["selected"] or
                not np.allclose(best["validation"],expected["validation"],rtol=0,atol=TIE)):
            raise ValueError("Cold development seed did not reproduce its frozen result")
        progress("final_test")
        start = time.monotonic()
        paths = PreparedSplitPaths(dataset,seed,row["target"],
                                   *(source/row[f"{s}_relpath"] for s in ("train","val","test")))
        job = JobSpec(identity,row["protocol_hash"],"removal",dataset,seed,"piv1_"+recipe,
                      {"columns":best["selected"]},(),"result.json")
        audit = AuditLog.from_manifest(out/"test_access.json",[job])
        test, ye = AuditedSplitAccess(paths, stage="removal", job_id=identity, audit=audit).load_test(
            reason="final_removal_evaluation", expected_sha256=partition_sha["test"])
        if list(test.columns) != names or len(ye) != row["test_row_count"]:
            raise ValueError("Unexpected locked test split")
        cols = best["selected"]
        pred = fit_predict(pd.concat([train[cols],val[cols]],ignore_index=True),np.concatenate([yt,yv]),
                           test[cols],**policy)
        fits += 1
        test_score = float(compute_utility(ye,pred,metric=metric))
        np.savez_compressed(out/"predictions.npz",y=ye,selected=pred)
        stages["final_s"] = time.monotonic()-start
    atomic_write_json(out/"result.json",dict(choices=winners, test_utility=test_score, full_validation=full_scores[0],
                      fits=fits, frozen_hash=identity, **stages))
    progress("complete")


def worker(*args):
    try:
        run_arm(*args)
    except Exception:
        (args[-2]/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


def run(phase, root, jobs, catalog, lock=None):
    sets = [(s,d) for s in ([42] if phase=="develop" else FINAL_SEEDS) for d in DATASETS]
    if phase=="final":
        output = root/"final"
    else:
        output = root/"development"
    output.mkdir(exist_ok=True)
    records = []
    with (output/"run.log").open("a", encoding="utf-8",buffering=1) as log_file:
        def live(message):
            print(message,flush=True)
            print(message,file=log_file,flush=True)

        for seed, dataset in sets:
            source = jobs[dataset,seed]
            with (source/"data_manifest.csv").open(encoding="utf-8-sig",newline="") as stream:
                row, = (r for r in csv.DictReader(stream) if r["dataset_id"]==dataset and int(r["seed"])==seed)
            for split in ("train","val","test"):
                row[f"{split}_row_count"] = int(row[f"{split}_row_count"])
            policy = dict(task=row["task"], seed=seed, model_params=catalog["shared"]["model_params"],
                          backend=catalog["shared"]["backend"])
            if phase=="develop":
                arms = [("shared",(*RECIPES,*DIAGNOSTICS))]
            else:
                methods = (lock["primary_recipe"],"anchor")
                if (DATASETS.index(dataset)+FINAL_SEEDS.index(seed))%2:
                    methods = methods[::-1]
                arms = [(m,(m,)) for m in methods]
            for label, recipes in arms:
                out = output / f"seed-{seed}" / dataset / label
                expected = None
                if phase=="final" and seed==42:
                    prior = read(root/"development"/"seed-42"/dataset/"shared"/"frozen.json")
                    if canonical_hash(prior) != lock["decisions"][dataset]:
                        raise ValueError("Validation decisions changed after locking the global recipe")
                    expected = prior["choices"][label]
                timer = read(out/"watchdog.json") if (out/"watchdog.json").is_file() else {}
                if timer.get("status") == "completed":
                    if (read(out/"result.json") != timer["result"] or
                            canonical_hash(read(out/"frozen.json")) != timer["result"]["frozen_hash"]):
                        raise ValueError(f"Completed arm changed: {out}")
                else:
                    if out.exists():
                        out.rename(out.with_name(out.name + ".incomplete-" + datetime.now(MSK).strftime("%H%M%S-%f")))
                    out.mkdir(parents=True)
                    live(f"{phase}/{dataset}/{seed}/{label} started; wall limit {LIMIT}s")
                    proc = multiprocessing.get_context("spawn").Process(
                        target=worker,args=(source,row,policy,recipes,phase,out,expected))
                    try:
                        timer = supervise(proc,out,LIMIT,live,heartbeat_s=10.,
                            heartbeat_formatter=lambda p,t,left: f"{dataset}/{label} {p.get('phase','starting')} "
                            f"{p.get('recipe','')} {p.get('checked',0)}/{p.get('total','?')} "
                            f"elapsed={t:.0f}s fits={p.get('fits',0)} search_ETA={p.get('eta_s',0):.0f}s "
                            f"deadline_in={left:.0f}s")
                    finally:
                        proc.close()
                if timer["status"]!="completed":
                    failure = (out/"error.log").read_text(encoding="utf-8") if (out/"error.log").exists() else ""
                    raise RuntimeError(f"{out}: {timer['status']}\n{failure}")
                result = timer["result"]
                for recipe, best in result["choices"].items():
                    records.append(dict(dataset=dataset,seed=seed,recipe=recipe,metric=row["utility"],
                        width=int(row["selected_feature_count"]),retained=best["retained"],
                        validation=best["validation"][0],test_utility=result["test_utility"],
                        fits=result["fits"],wall_s=timer["elapsed_s"],voting_s=result["voting_s"],
                        candidates=best["checked"],winner_source=best["source"]))
                write_summary(output,records)
                live(f"{dataset}/{seed}/{label} completed in {timer['elapsed_s']:.2f}s, fits={result['fits']}, "
                     f"validation={ {k:round(v['validation'][0],6) for k,v in result['choices'].items()} }")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("develop","final"))
    parser.add_argument("--run",type=Path,help="Completed development run to lock and evaluate")
    args = parser.parse_args()
    jobs,catalog = audit_prepared_catalog()
    hashes = source_hashes()
    if args.phase=="develop":
        if args.run:
            parser.error("Development always creates a fresh run directory.")
        root = OUTPUT / datetime.now(MSK).strftime("PIv1-pilot-%Y%m%d-%H%M%S-%f")
        root.mkdir(parents=True)
        protocol = dict(version="PIv1",catalog_hash=catalog["sha256"],source_hashes=hashes,
                        datasets=DATASETS, development_seed=42, replication_seed=123, methods=RECIPES,
                        development_only_mean_ablations=DIAGNOSTICS,
                        global_selection="Sum of within-dataset validation ranks across the two voting recipes. "
                        "Equal utility within 1e-14 shares the rank. Aggregate tie favors vote.",
                        common_limit_s_per_arm=LIMIT, prior_run=str(BASELINE))
        atomic_write_json(root/"protocol.json",protocol)
        (root/"source").mkdir()
        for path in source_files():
            shutil.copy2(path,root/"source"/path.name)
        print(f"Output: {root}",flush=True)
        with exclusive_run(root):
            run("develop",root,jobs,catalog)
    else:
        if not args.run:
            parser.error("Final evaluation requires --run with the development directory.")
        root = args.run.resolve()
        protocol = read(root/"protocol.json")
        if hashes!=protocol["source_hashes"] or protocol["catalog_hash"]!=catalog["sha256"]:
            raise ValueError("Code or prepared data changed since the development run")
        decisions = {d:read(root/"development"/"seed-42"/d/"shared"/"frozen.json") for d in DATASETS}
        chosen, ranks = choose_recipe([v["choices"] for v in decisions.values()])
        lock = dict(primary_recipe=chosen, ablation="anchor", ranks=ranks, seeds=FINAL_SEEDS,
                    protocol_hash=canonical_hash(protocol), decisions={k:canonical_hash(v) for k,v in decisions.items()})
        with exclusive_run(root):
            if (root/"lock.json").exists():
                if canonical_hash(read(root/"lock.json")) != canonical_hash(lock):
                    raise ValueError("Global lock changed")
            else:
                atomic_write_json(root/"lock.json",lock)
            print(f"Locked global recipe: {chosen}, ranks={ranks}",flush=True)
            run("final",root,jobs,catalog,lock)
    return 0


if __name__=="__main__":
    raise SystemExit(main())

