"""Compare revised document minPI rules on fixed development and validation folds."""
import argparse
from contextlib import nullcontext
import importlib
import inspect
import json
import platform
import re
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from experiments import document_vote_benchmark as original
from experiments import mail_vote_transfer as mail
from experiments import refined_document_external as external
from experiments.fast_document_vote_benchmark import pooled_binary_gain
from experiments.fast_minpi_five_source import check_columns, summarize_timings
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training, library_path


ROOT = original.ROOT
KEY = original.CASE
DEVELOPMENT = {("sms", 1), ("amazon_cells", 1), ("imdb", 1), ("yelp", 1)}
SHARED_ARMS = ("old_vote", "F50", "chi50", "pooled50", "random50", "full")
REFERENCE = ROOT / "results/minpi-study/fast-document-20261117a"
PROTOCOL = ROOT / "docs/minpi-study/refined-document-protocol.md"


def load_candidates(specifications):
    candidates = {}
    for specification in specifications:
        name, target = specification.split("=", 1)
        module, member = target.rsplit(":", 1)
        if not re.fullmatch("[a-z][a-z0-9]*", name) or name in SHARED_ARMS or name in candidates:
            raise ValueError("Candidate names must be unique lowercase letters/digits")
        candidates[name] = getattr(importlib.import_module(module), member)
    return candidates


def methods(candidates):
    arms = []
    for name in candidates:
        arms.extend((name, "F_" + name, "pooled_" + name, "random_" + name))
    return tuple(arms) + SHARED_ARMS


def panels(stage):
    if stage == "external":
        yield from external.panels()
        return
    for X, y, Xt, truth, info in original.panels():
        split = "dev" if (info["dataset"], info["fold"]) in DEVELOPMENT else "validation"
        if stage in ("all", split):
            yield X, y, Xt, truth, dict(info, suite="uci", stage=split)
    if stage != "dev":
        X, y, Xt, truth, info = mail.prepare()
        for seed in mail.SEEDS:
            yield X, y, Xt, truth, dict(info, seed=seed, suite="mail", stage="validation")


def select(arm, X, y, seed, widths, candidates):
    if arm in candidates:
        return candidates[arm](X, y, random_state=seed)
    if arm == "old_vote":
        return fast_document_vote_top_from_training(X, y, random_state=seed, backend="native")
    if arm == "full":
        return np.arange(X.shape[1]), {}
    if arm.endswith("50"):
        rule, count = arm[:-2], (X.shape[1] + 1) // 2
    else:
        rule, candidate = arm.split("_", 1)
        count = widths[candidate]
    if rule == "pooled":
        score = pooled_binary_gain(X, y)
        return np.sort(np.argsort(-score, kind="stable")[:count]), {}
    return original.select(rule + "_match", X, y, seed, count), {}


def snapshot(candidates, args):
    files = {Path(__file__), PROTOCOL, Path(original.__file__), Path(mail.__file__), Path(external.__file__),
             ROOT / "experiments/fast_document_vote_benchmark.py",
             ROOT / "experiments/fast_minpi_five_source.py",
             ROOT / "experiments/minpi_constant_transfer.py",
             ROOT / "experiments/compare_predictive_minpi.py",
             ROOT / "experiments/minpi_study.py", ROOT / "configs/minpi-study.json",
             library_path()}
    # Record the loaded project helpers as well as each selected candidate.
    for name, module in list(sys.modules.items()):
        if name.startswith("minppi_mofs.") and getattr(module, "__file__", None):
            files.add(Path(module.__file__))
            source = getattr(module, "SOURCE", None)
            if isinstance(source, Path) and source.is_file():
                files.add(source)
    files.update(Path(inspect.getsourcefile(function)) for function in candidates.values())
    files.update(Path(p).resolve() for p in args.source)
    return dict(
        sources={p.relative_to(ROOT).as_posix(): original.file_hash(p) for p in sorted(files)},
        candidates=args.candidate, methods=list(methods(candidates)), stage=args.stage,
        independent_pass=args.pass_id, repeats=len(methods(candidates)), threads=1,
        development=sorted([list(item) for item in DEVELOPMENT]),
        uci_assets=original.ASSETS, mail_assets=mail.ARCHIVES, external_asset=external.SHA256,
        versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
        previous_protocol_sha256=original.file_hash(REFERENCE / "protocol.json"),
    )


def run_case(X, y, Xt, truth, info, candidates, args):
    case = {key: info[key] for key in KEY}
    seed = info["seed"]
    filename = f"{info['dataset']}-{info['fold']}-{seed}.npz"
    arrays, widths, scores, audits, trials = {}, {}, {}, [], []
    arms = methods(candidates)
    archive = np.load(REFERENCE / filename) if info["suite"] != "external" else nullcontext(None)
    with archive as previous:
        for arm in arms:
            cols, diagnostics = select(arm, X, y, seed, widths, candidates)
            check_columns(cols, X.shape[1])
            if arm in candidates:
                widths[arm] = len(cols)
                for field in ("scores", "zero_peer_scores", "one_peer_scores"):
                    if field in diagnostics:
                        value = np.asarray(diagnostics[field])
                        assert value.shape == (X.shape[1],) and np.isfinite(value).all()
                        arrays[arm + "_" + field] = value
                if "replicate_endpoints" in diagnostics:
                    arrays[arm + "_replicate_endpoints"] = np.asarray(diagnostics["replicate_endpoints"])
            elif "_" in arm and arm != "old_vote":
                assert len(cols) == widths[arm.split("_", 1)[1]]
            pred = original.fit_predict(X, y, Xt, cols, "classification", seed, False)
            scores[arm] = original.balanced_accuracy_score(truth, pred)
            arrays[arm + "_cols"], arrays[arm + "_pred"] = cols, pred
            if arm in SHARED_ARMS and previous is not None:
                old_name = {"old_vote": "fast_vote", "full": "full"}.get(arm, arm[:-2] + "_50")
                np.testing.assert_array_equal(cols, previous[old_name + "_cols"])
                np.testing.assert_array_equal(pred, previous[old_name + "_pred"])
            audits.append(dict(**case, method=arm, width=len(cols), score=scores[arm],
                               backend=diagnostics.get("backend", "baseline"),
                               support_sha256=original.digest(cols), prediction_sha256=original.digest(pred)))
    if args.mode == "verify":
        with np.load(args.output / filename) as saved:
            assert set(saved.files) == set(arrays)
            for field, value in arrays.items():
                np.testing.assert_array_equal(value, saved[field], err_msg=filename + " " + field)
        return trials, audits
    np.savez_compressed(args.output / filename, **arrays)
    order = np.random.default_rng([730493, args.pass_id, info["fold"], seed]).permutation(arms)
    for repeat in range(len(arms)):
        for position, arm in enumerate(np.roll(order, repeat)):
            start = perf_counter_ns()
            cols, _ = select(arm, X, y, seed, widths, candidates)
            selected = perf_counter_ns()
            pred = original.fit_predict(X, y, Xt, cols, "classification", seed, False)
            finish = perf_counter_ns()
            np.testing.assert_array_equal(cols, arrays[arm + "_cols"])
            np.testing.assert_array_equal(pred, arrays[arm + "_pred"])
            trials.append(dict(**case, method=arm, repeat=repeat, position=position,
                               width=len(cols), p=X.shape[1], score=scores[arm],
                               select_ms=(selected-start)/1e6, refit_ms=(finish-selected)/1e6,
                               total_ms=(finish-start)/1e6))
    return trials, audits


def summarize(output, trials, panels_frame, candidates):
    cases = summarize_timings(output, trials)
    panels_frame.to_csv(output / "panels.csv", index=False)
    comparisons, votes = [], []
    for name in candidates:
        vote = cases[cases.method == name].set_index(KEY).join(
            panels_frame.set_index(KEY)[["suite", "stage", "task", "common_prepare_ms"]])
        vote["candidate"] = name
        vote["ratio_with_vocabulary"] = (
            (vote.total_ms + vote.common_prepare_ms) / (vote.full_total_ms + vote.common_prepare_ms))
        refs = (*SHARED_ARMS, "F_" + name, "pooled_" + name, "random_" + name)
        pairs = []
        for ref in refs:
            control = cases[cases.method == ref].set_index(KEY)
            comp = vote[["candidate", "suite", "stage", "task", "joint"]].copy()
            comp["baseline"] = ref
            comp["score_delta"] = vote.score - control.score
            comp["time_ratio"] = vote.total_ms / control.total_ms
            comp["width_ratio"] = vote.width / control.width
            safe = vote.joint & (comp.score_delta >= -.02)
            comp["relative_joint"] = safe & (comp.time_ratio < 1) & (comp.width_ratio <= 1)
            comp["relative_joint_10pct_width"] = safe & (comp.time_ratio < 1) & (comp.width_ratio <= 1.1)
            comp["relative_joint_10pct_time"] = safe & vote.joint_10pct & (comp.time_ratio <= .9) & (comp.width_ratio <= 1)
            pairs.append(comp)
        paired = pd.concat(pairs).reset_index()
        for field in ("relative_joint", "relative_joint_10pct_width", "relative_joint_10pct_time"):
            flags = paired.pivot(index=KEY, columns="baseline", values=field)
            vote["half_" + field] = flags[["F50", "chi50", "pooled50", "random50"]].all(axis=1)
            vote["matched_supervised_" + field] = flags[["F_" + name, "pooled_" + name]].all(axis=1)
            vote["matched_all_" + field] = flags[["F_" + name, "pooled_" + name, "random_" + name]].all(axis=1)
        comparisons.append(paired)
        votes.append(vote.reset_index())
    paired = pd.concat(comparisons, ignore_index=True)
    vote = pd.concat(votes, ignore_index=True)
    paired.to_csv(output / "paired-baselines.csv", index=False)
    vote.to_csv(output / "vote-comparisons.csv", index=False)
    sums = dict(configs=("fold", "size"), joint=("joint", "sum"), joint_10pct=("joint_10pct", "sum"),
                half=("half_relative_joint", "sum"), half_10pct=("half_relative_joint_10pct_time", "sum"),
                matched_supervised=("matched_supervised_relative_joint", "sum"),
                matched_all=("matched_all_relative_joint", "sum"), mean_score=("score", "mean"),
                median_width=("width", "median"), median_ratio=("ratio", "median"),
                worst_gap=("gap", "min"))
    vote.groupby(["candidate", "suite", "stage"]).agg(**sums).to_csv(output / "candidate-summary.csv")
    vote.groupby(["candidate", "dataset"]).agg(**sums).to_csv(output / "candidate-by-source.csv")


def random_draws(candidates, args):
    cases = pd.read_csv(args.output / "cases.csv").set_index(KEY + ["method"])
    draws = []
    for X, y, Xt, truth, info in panels(args.stage):
        case = {key: info[key] for key in KEY}
        for name in candidates:
            point = cases.loc[(*[case[key] for key in KEY], name)]
            width = int(point.width)
            for draw in range(20):
                rng = np.random.default_rng([812381, info["fold"], info["seed"], draw])
                cols = np.sort(rng.permutation(X.shape[1])[:width])
                pred = original.fit_predict(X, y, Xt, cols, "classification", info["seed"], False)
                score = original.balanced_accuracy_score(truth, pred)
                draws.append(dict(**case, candidate=name, draw=draw, width=width, score=score,
                                  vote_score=point.score, win=point.score > score + 1e-12,
                                  support_sha256=original.digest(cols), prediction_sha256=original.digest(pred)))
    frame = pd.DataFrame(draws)
    path = args.output / "random-draws.csv"
    if path.exists():
        pd.testing.assert_frame_equal(frame, pd.read_csv(path), rtol=0, atol=1e-12)
        original.write_json(args.output / "random-replay.json",
                            dict(rows=len(frame), sha256=original.file_hash(path), exact_supports=True))
    else:
        frame.to_csv(path, index=False)
    frame.groupby(["candidate", *KEY]).agg(
        width=("width", "first"), score=("vote_score", "first"), random_wins=("win", "sum"),
        random_median=("score", "median"), random_best=("score", "max")
    ).to_csv(args.output / "random-comparisons.csv")


def run(args):
    candidates = load_candidates(args.candidate)
    state = snapshot(candidates, args)
    if args.mode == "measure":
        args.output.mkdir(parents=True, exist_ok=False)
        original.write_json(args.output / "protocol.json", state)
    else:
        saved = json.loads((args.output / "protocol.json").read_text(encoding="utf8"))
        assert state == saved, "Source, dataset or measurement configuration changed"
    if args.mode == "random":
        random_draws(candidates, args)
        assert snapshot(candidates, args) == state
        return
    trials, audits, metadata = [], [], []
    for X, y, Xt, truth, info in panels(args.stage):
        times, audit = run_case(X, y, Xt, truth, info, candidates, args)
        trials.extend(times)
        audits.extend(audit)
        metadata.append(info)
        print(args.mode, info["dataset"], info["fold"], info["seed"], flush=True)
    expected = {"dev": 4, "validation": 16, "all": 20, "external": 3}[args.stage]
    assert len(metadata) == expected and len(audits) == expected * len(methods(candidates))
    frame = pd.DataFrame(metadata)
    assert snapshot(candidates, args) == state, "Source changed while the experiment was running"
    if args.mode == "verify":
        fields = frame.columns.drop("common_prepare_ms")
        pd.testing.assert_frame_equal(
            frame[fields], pd.read_csv(args.output / "panels.csv", keep_default_na=False)[fields])
        pd.DataFrame(audits).to_csv(args.output / "fresh-process-audit.csv", index=False)
    else:
        assert len(trials) == expected * len(methods(candidates)) ** 2
        summarize(args.output, trials, frame, candidates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=module:callable")
    parser.add_argument("--stage", choices=("dev", "validation", "all", "external"), default="dev")
    parser.add_argument("--mode", choices=("measure", "verify", "random"), default="measure")
    parser.add_argument("--pass-id", type=int, choices=(1, 2), default=1)
    parser.add_argument("--source", action="append", default=[], help="Additional implementation source or native library")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args)
