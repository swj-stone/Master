"""Replay the 18 additional low-dimensional UCI cases in the historical 405-case panel."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from research.block_selection.fresh import fresh_cases
from research.conditional_dense.original_tasks.paper_splits import check_f
from research.fast_single import runner, selector, summarize as report
from research.pair_selection.benchmark import external_cases
from research.wide_blocks.panels import Case, KEY, bench, input_hashes


HISTORY = Path(__file__).resolve().parents[2]
EXTENDED = HISTORY / "research/wide_blocks/evidence/all-paired.csv"
ARCHIVE = bench.ARCHIVE / "block-completion-20261119a"
SUITES = ("external", "fresh")


def registry():
    extended = pd.read_csv(EXTENDED, keep_default_na=False, float_precision="round_trip").set_index(KEY)
    main = pd.read_csv(report.PAPER, usecols=KEY, keep_default_na=False).set_index(KEY)
    extra = extended.loc[~extended.index.isin(main.index)].copy()
    if (len(extended) != 405 or len(main) != 387 or not main.index.isin(extended.index).all()
            or len(extra) != 18 or not extra.index.is_unique or not extra.p.le(32).all()
            or set(extra.index.get_level_values("cohort")) != set(SUITES)):
        raise ValueError("Historical 405-case extension differs")
    extra["case_id"] = np.arange(387, 405)
    return extra


def output_stats(case, cols):
    pred, decision = case.predict(cols)
    score = float(case.score(pred, decision))
    hashes = {label: bench.digest(a) if a is not None else "" for label, a in zip(
        runner.replay.HASHES, (cols, pred, decision))}
    return (cols, pred, decision), dict(k=len(cols), score=score, **hashes)


def references(case, old, quality):
    # The old experiments saved explicit supports and per-sample output digests.
    # Rebuild the arrays solely to feed the exact pass-1/pass-2 timing harness.
    fields, ref = {}, {}
    single_width = int(quality.at["current", "k"])
    for arm, archived in (("full", "full"), ("single", "current"), ("F_single", "F"), ("random_single", "random")):
        row = quality.loc[archived]
        stored_width = arm in ("single", "full") or row.k == single_width
        if stored_width:
            cols = np.fromstring(str(row.cols), dtype=np.int64, sep=" ")
        elif arm == "F_single":
            cols = case.select_F(single_width)
        else:
            cols = np.sort(np.random.default_rng(case.seed).permutation(old.p)[:single_width])
        if arm == "F_single":
            check_f(case.selection_X(), case.y, case.task, cols)
        arrays, stats = output_stats(case, cols)
        if stored_width:
            if (len(cols) != row.k or stats["pred_sha"] != row.prediction_sha
                    or stats["decision_sha"] != row.decision_sha):
                raise ValueError(f"Archived outputs differ on {case.key}, {arm}")
            np.testing.assert_allclose(stats["score"], row.score, rtol=0, atol=2e-12)
        for field, value in stats.items():
            ref[f"{field}_{arm}"] = value
        if arm != "single":
            for suffix, value in zip(("cols", "pred", "decision"), arrays):
                if value is not None:
                    fields[f"{archived}_{suffix}"] = value
    np.savez(case.archive, **fields)
    old = old.copy()
    old["k_new"] = single_width
    for kind in ("full", "F", "random"):
        old[f"score_{kind}"] = ref["score_" + (kind if kind == "full" else kind + "_single")]
    return old, pd.Series(dict(case_id=old.case_id, **ref))


def run(folder, process):
    if any(pool["num_threads"] != 1 for pool in threadpool_info()):
        raise ValueError("Single-threaded numerical runtimes required")
    manifest = registry()
    protocol = runner.protocol(3, selector)
    files = [Path(__file__), EXTENDED, HISTORY/"research/pair_selection/external.py",
             HISTORY/"research/block_selection/fresh.py",
             *(ARCHIVE/f"{s}-pass1"/"quality.csv" for s in SUITES)]
    protocol["extension_hashes"] = {p.relative_to(HISTORY).as_posix(): bench.checksum(p) for p in files}
    if (folder/"protocol.json").exists():
        if json.loads((folder/"protocol.json").read_text()) != protocol:
            raise ValueError("Source or protocol changed; use a fresh output directory")
    elif process == 1:
        folder.mkdir(parents=True, exist_ok=True)
        (folder/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf8")
    else:
        raise ValueError("Pass 1 is required")
    target = folder / f"pass{process}"
    target.mkdir(exist_ok=True)
    qualities = pd.concat([pd.read_csv(ARCHIVE/f"{s}-pass1"/"quality.csv", keep_default_na=False,
                                       float_precision="round_trip") for s in SUITES]).set_index([*KEY, "method"])
    checked = []
    with TemporaryDirectory(dir=folder) as cache:
        for original in [*external_cases(), *fresh_cases()]:
            key = original.key
            old = manifest.loc[key]
            case = Case(key, original.X, original.y, original.Xt, original.truth, original.task, old.metric,
                        Path(cache)/"controls.npz", "training_means" if original.impute else "",
                        original.predict, original.score)
            if (original.X.shape != (old.n_train, old.p) or original.Xt.shape != (old.n_test, old.p)
                    or input_hashes(case) != tuple(old[report.HASHES])):
                raise ValueError(f"Original inputs differ on {key}")
            old, ref = references(case, old, qualities.loc[key])
            out = target/f"case-{int(old.case_id):03}.csv"
            if not out.exists():
                runner.run_case(case, old, ref, out, process, 3, selector.select_fast)
            checked.append(dict(zip(KEY, key), **old[["task", "metric", "p", "source", "n_train", "n_test",
                                                    *report.HASHES]].to_dict(), **ref.to_dict()))
    reference = pd.DataFrame(checked).set_index("case_id").sort_index()
    if not reference.index.is_unique or set(reference.index) != set(manifest.case_id):
        raise ValueError("Missing or repeated extension cases")
    first = folder/"reference.csv"
    if process == 1:
        reference.to_csv(first)
    else:
        stored = pd.read_csv(first, keep_default_na=False, float_precision="round_trip").set_index("case_id")
        pd.testing.assert_frame_equal(reference, stored, check_exact=True)
    (target/"complete.json").write_text(json.dumps(dict(process=process, cases=18, repeats=3)), encoding="utf8")
    print(f"Verified {len(reference)} historical extension cases in pass {process}")


def summarize(folder, main_path, out):
    expected = pd.read_csv(folder/"reference.csv", keep_default_na=False, float_precision="round_trip").set_index("case_id")
    if set(expected.index) != set(registry().case_id):
        raise ValueError("Incomplete extension reference")
    results = []
    for i, row in expected.iterrows():
        passes = [report.read_case(folder/f"pass{idx}"/f"case-{i:03}.csv", i, idx, row, 3) for idx in (1, 2)]
        pd.testing.assert_frame_equal(passes[0][report.STABLE], passes[1][report.STABLE], check_exact=True)
        s = passes[0]
        if not s.loc["fast", report.STABLE].equals(s.loc["single", report.STABLE]):
            raise ValueError(f"Low-dimensional rule changed outputs on case {i}")
        record = row[[*report.KEY, "source", "n_train", "n_test", *report.HASHES]].to_dict()
        for arm in report.ARMS:
            record.update({f"{field}_{arm}": s.at[arm, field] for field in report.STABLE})
            for idx, frame in enumerate(passes, 1):
                record.update({f"{field}{idx}_{arm}": frame.at[arm, field] for field in ("select_ms", "total_ms")})
        results.append(dict(case_id=i, **record))
    extra = pd.DataFrame(results).set_index("case_id")
    main = pd.read_csv(main_path, keep_default_na=False, float_precision="round_trip").set_index("case_id")
    union = pd.concat([main, extra], verify_integrity=True)
    if (len(union) != 405 or union.duplicated(KEY).any()
            or set(union.set_index(KEY).index) != set(pd.read_csv(EXTENDED, usecols=KEY, keep_default_na=False).set_index(KEY).index)):
        raise ValueError("405-case union disagrees with historical panel")
    out.mkdir(parents=True, exist_ok=True)
    union.to_csv(out/"all-405.csv")
    rows = []
    for name, subset in [("论文面板", main), ("额外低维", extra), ("历史全部", union), *extra.groupby("dataset")]:
        for arm in report.ARMS:
            if arm == "full":
                continue
            c = report.compare(subset, arm, "full")
            d = report.summarize(c)
            owner = "single" if "single" in arm else "fast"
            r = report.compare(subset, arm, f"random_{owner}")
            rows.append(dict(scope=name, arm=arm, **d, random_ratio=r.ratio.median(),
                             cut_faster_random=int((r.cut & r.fast).sum()),
                             cut_near_random=int((r.cut & r.ratio.le(1.10)).sum())))
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"by-source.csv", index=False)
    lines = ["# 历史 405 组的补充复测", "", "主面板 387 组和另行测量的 18 组在同一口径下合并。补充面板含六个 UCI 数据集，每个三折，输入均不超过 32 列。候选在这一区间沿用论文选择规则，两个选择器的列、预测和得分已逐组核对。", "",
             "两进程均须更快，允许 0.02 表示相对 Full 的原指标绝对损失。随机时间比取两次配对比值中的较大值，再对配置取中位数。5% 或 10% 等容差用于描述接近程度。", ""]
    view = frame[frame.arm.isin(report.RULES)]
    lines.append(report.table(["范围", "方法", "配置", "保留 %", "对 Full 删列加速", "再要求损失≤0.02",
                               "对同宽随机时间比", "快于随机", "≤1.10倍随机"],
                              ([r.scope, report.NUMBERS[r.arm], r.n, f"{r.median_retained_pct:.1f}",
                                r.cut_faster, r.joint_002, f"{r.random_ratio:.2f}", r.cut_faster_random,
                                r.cut_near_random] for r in view.itertuples())))
    lines.extend(["", "[405 组原始配对明细](all-405.csv) · [六种选列方法的分组结果](by-source.csv)", ""])
    (out/"summary.md").write_text("\n".join(lines), encoding="utf8")
    print(f"Verified 405 configurations, including {len(extra)} historical additional cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--process", type=int, choices=(1, 2))
    choice.add_argument("--summarize", action="store_true")
    parser.add_argument("--main", type=Path, default=report.HERE/"evidence/paired-cases.csv")
    parser.add_argument("--out", type=Path, default=report.HERE/"evidence-extension")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.folder, args.main, args.out)
    else:
        with threadpool_limits(1):
            run(args.folder, args.process)
