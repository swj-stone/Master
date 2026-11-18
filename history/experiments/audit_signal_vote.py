"""Recalculate vote vs same-width evidence from individual timed trials and random draws."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "fold", "seed"]


def checked_rows(run, *, confirmation=False, peer_simulation=False):
    vote, marginal, f, random = (("vote", "zero_peer", "F_vote", "random_vote") if confirmation else
                                 ("signal_auto", "no_peer", "F_signal", "random_signal"))
    raw = pd.read_csv(run / "trials.csv")
    group = [*KEY, "method"]
    assert not raw.duplicated([*group, "repeat"]).any()
    assert raw.groupby(group)[["width", "p", "score"]].nunique().eq(1).all().all()
    assert raw.groupby(group).size().isin([5, 7]).all()
    med = raw.groupby(group, as_index=False).agg(width=("width", "first"), p=("p", "first"),
        score=("score", "first"), select_ms=("select_ms", "median"), total_ms=("total_ms", "median"))
    replay = pd.read_csv(run / "fresh-process-audit.csv").set_index(group).sort_index()
    pd.testing.assert_index_equal(med.set_index(group).sort_index().index, replay.index)
    for col in ["prediction_sha256", "support_sha256"]:
        assert replay[col].str.fullmatch("[0-9a-f]{64}").all()
    full = med[med.method == "full"].set_index(KEY)
    rows = med[med.method == vote].set_index(KEY).copy()
    rows["full_score"] = full.score
    rows["gap"] = rows.score - full.score
    rows["keep_fraction"] = rows.width / rows.p
    rows["ratio"] = rows.total_ms / full.total_ms
    for arm, name in [(marginal, "zero"), (f, "F"), (random, "random")]:
        control = med[med.method == arm].set_index(KEY)
        pd.testing.assert_series_equal(control.width, rows.width)
        rows["delta_" + name] = rows.score - control.score
        rows[name + "_joint"] = ((control.width < rows.p) & (control.score - full.score >= -.02) &
                                 (control.total_ms < full.total_ms))
    draws = pd.read_csv(run / "random-draws.csv")
    assert not draws.duplicated([*KEY, "draw"]).any()
    draws = draws.merge(rows[["width", "score"]], left_on=KEY, right_index=True, suffixes=("", "_checked"))
    score_name = "random_score" if "random_score" in draws else "score"
    vote_score = "score_checked" if "score_checked" in draws else "score"
    assert draws.width.eq(draws.width_checked).all()
    draws["win_against_draw"] = draws[vote_score] > draws[score_name] + 1e-12
    draw_groups = draws.groupby(KEY)
    rows["draws"] = draw_groups.size()
    assert rows.draws.isin([20, 32]).all()
    rows["random_wins"] = draw_groups.win_against_draw.sum()
    rows["random_median"] = draw_groups[score_name].median()
    rows["better_than_random_median"] = rows.score > rows.random_median + 1e-12
    rows["reduced"] = rows.width < rows.p
    rows["quality"] = rows.gap >= -.02
    rows["joint"] = rows.reduced & rows.quality & (rows.ratio < 1)
    rows["joint10"] = rows.reduced & rows.quality & (rows.ratio <= .90)
    rows["strong_joint"] = rows.joint & (4 * rows.random_wins >= 3 * rows.draws)
    rows["joint_no_loss"] = rows.reduced & (rows.gap >= 0) & (rows.ratio < 1)
    saved = pd.read_csv(run / "cases.csv")
    saved = saved[saved.method == vote].set_index(KEY)
    for field in ["joint", "width", "p"]:
        np.testing.assert_array_equal(rows[field].sort_index(), saved[field].sort_index())
    if peer_simulation:
        from experiments.signal_vote_peer_mechanism import N_TRAIN
        rows["n_train"] = N_TRAIN
        rows["task"] = [name.split("-")[0] for name in rows.index.get_level_values("dataset")]
    else:
        meta = pd.read_csv(run / "panels.csv").set_index(KEY)
        rows["n_train"] = meta.n_train
        rows["task"] = meta.task if "task" in meta else "classification"
    rows["concentrated"] = (rows.n_train >= 512) & (rows.p >= 32) & (rows.width * 3 <= rows.p)
    rows["run"] = run.name
    return rows.reset_index()


def main(args):
    data = pd.concat([checked_rows(folder, confirmation=(folder in (args.confirmation, args.real_proxies)),
                                   peer_simulation=(folder == args.peer))
                      for folder in [args.original, args.transfer, args.confirmation, args.sources, args.peer,
                                     args.real_proxies]])
    assert list(data.groupby("run", sort=False).size()) == [69, 57, 20, 8, 20, 12]
    args.output.mkdir(parents=True, exist_ok=False)
    data.to_csv(args.output / "comparisons.csv", index=False)
    for keys, name in [(["run"], "summary"), (["run", "dataset"], "by-source"),
                       (["run", "concentrated"], "training-condition")]:
        data.groupby(keys).agg(
            configs=("width", "size"), sources=("dataset", "nunique"), reduced=("reduced","sum"),
            quality=("quality","sum"), joint=("joint","sum"), strict=("strong_joint","sum"),
            joint10=("joint10","sum"), joint_no_loss=("joint_no_loss","sum"), random_joint=("random_joint","sum"),
            F_joint=("F_joint","sum"), marginal_joint=("zero_joint","sum"),
            median_random_wins=("better_than_random_median","sum"),
            vote_beats_F=("delta_F", lambda x: (x > 1e-12).sum()),
            vote_ties_F=("delta_F", lambda x: (x.abs() <= 1e-12).sum()),
            vote_beats_marginal=("delta_zero", lambda x: (x > 1e-12).sum()),
            vote_ties_marginal=("delta_zero", lambda x: (x.abs() <= 1e-12).sum()),
            median_keep=("keep_fraction","median"), median_ratio=("ratio","median"),
            median_select_ms=("select_ms","median")).to_csv(args.output / (name + ".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["original", "transfer", "confirmation", "sources", "peer", "real-proxies", "output"]:
        parser.add_argument("--" + arg, required=True, type=Path)
    main(parser.parse_args())
