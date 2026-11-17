"""Observed monitor outages on a synchronized, embargoed Beijing time split."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import predictive_selection as select
from experiments.minpi_rework import quota_coalition_rf as rf
from experiments.minpi_rework import validation_predictive as primary
from experiments.minpi_rework.predictive_guard import margin_path
from experiments.minpi_rework.quota_compress_tree import Game
from minppi_mofs.predictive_quota import PredictiveGame


OUT = b.ROOT / "thesis-paper/evidence/minpi_rework/observed-availability"
NAMES = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
         "wind_sin", "wind_cos"]
SEED = 42


def data():
    source = primary.SOURCES["beijing"]
    raw = (primary.DATA / source["filename"]).read_bytes()
    assert sha256(raw).hexdigest() == source["sha256"]
    compass = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()
    wind = {angle: i * np.pi / 8 for i, angle in enumerate(compass)}
    frames = []
    with ZipFile(BytesIO(raw)) as outer, ZipFile(BytesIO(outer.read("PRSA2017_Data_20130301-20170228.zip"))) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
            hourly = pd.read_csv(z.open(name))
            f = hourly.loc[hourly.hour.eq(12)].copy()
            f.index = pd.to_datetime(f[["year", "month", "day"]])
            assert f.index.is_unique and f.index.is_monotonic_increasing and f.station.nunique() == 1
            direction = f.wd.map(wind)
            assert direction.isna().equals(f.wd.isna())
            f["wind_sin"], f["wind_cos"] = np.sin(direction), np.cos(direction)
            f["target"] = f["PM2.5"].reindex(f.index + pd.Timedelta(days=1)).to_numpy()
            f["date"] = f.index
            frames.append(f[["date", "station", "target", *NAMES]])
    frame = pd.concat(frames, ignore_index=True).sort_values(["date", "station"], ignore_index=True)
    days = frame.date.to_numpy("datetime64[D]")
    dates = np.unique(days)
    assert frame.station.nunique() == 12 and (np.diff(dates).astype(int) == 1).all()
    assert len(frame) == 12 * len(dates) and not frame.duplicated(["date", "station"]).any()
    cuts = dates[[int(f * len(dates)) for f in (.48, .64, .8)]]
    slot = np.searchsorted(cuts, days, side="right")
    embargo = np.searchsorted(cuts, days + np.timedelta64(1, "D"), side="right") != slot
    available = frame.target.notna().to_numpy()
    records = dict(stations=12, origins=len(frame), missing_response=int((~available).sum()),
                   embargo_with_response=int((embargo & available).sum()), cuts=cuts.astype(str).tolist())
    frame["block"] = np.asarray(primary.PARTS)[slot]
    frame = frame.loc[available & ~embargo].reset_index(drop=True)
    for first, second in zip(primary.PARTS, primary.PARTS[1:]):
        assert frame.loc[frame.block == first, "date"].max() + pd.Timedelta(days=1) < frame.loc[frame.block == second, "date"].min()
    return frame, records


def observed_game(game, x):
    """Restrict each row to its recorded columns before sampling further global masks."""
    bits = 1 << np.arange(game.p)
    unavailable = np.isnan(x).astype(np.int64) @ bits
    required = np.asarray(game.incidence @ bits).ravel().astype(np.int64)
    entries = game.a.tocoo()
    ok = (unavailable[entries.row] & required[entries.col]) == 0
    a = sparse.csc_matrix((entries.data[ok], (entries.row[ok], entries.col[ok])), shape=entries.shape)
    return Game(a, game.incidence)


def predict_observed(model, x, support):
    available = np.isfinite(x)
    selected = np.zeros(model.p, bool)
    selected[support] = True
    patterns, groups = np.unique(available & selected, axis=0, return_inverse=True)
    result = np.empty(len(x))
    for group, mask in enumerate(patterns):
        rows, cols = np.flatnonzero(groups == group), np.flatnonzero(mask)
        result[rows] = model.predict(x[np.ix_(rows, cols)], cols)
    return result


def exact_check(model, x, y):
    """Compare all masks against direct per-row traversal, including a total outage."""
    rows = [np.flatnonzero(np.isnan(x).any(axis=1))[0], np.flatnonzero(np.isfinite(x).all(axis=1))[0], 0]
    sample, labels = x[rows].copy(), y[rows]
    sample[-1] = np.nan
    restricted = observed_game(model.game(sample), sample)
    game = PredictiveGame.from_paths(restricted.a, restricted.incidence, labels - model.empty[0])
    p = model.p
    keep = np.arange(p) % 2 == 0
    totals, counts = np.zeros(p + 1), np.zeros(p + 1)
    for mask in range(1 << p):
        avail = ((mask >> np.arange(p)) & 1).astype(bool)
        size = int(avail.sum())
        full = predict_observed(model, sample, np.flatnonzero(avail))
        reduced = predict_observed(model, sample, np.flatnonzero(avail & keep))
        totals[size] += np.sum((labels - reduced) ** 2 - (labels - full) ** 2)
        counts[size] += 1
    error = np.abs(totals / counts - game.conditional_risk(keep)).max()
    np.testing.assert_allclose(totals / counts, game.conditional_risk(keep), rtol=1e-10, atol=1e-7)
    return dict(masks=1 << p, rows=len(sample), max_absolute_error=float(error))


def collect():
    frame, receipt = data()
    x = frame[NAMES].to_numpy(float)
    target = frame.target.to_numpy(float, copy=True)
    parts = {name: np.flatnonzero(frame.block.eq(name)) for name in primary.PARTS}
    fit, vote, width, test = (parts[k] for k in primary.PARTS)
    target[test] = np.nan
    assert np.isnan(target[test]).all()
    model = rf.fit(x[fit], target[fit], "regression", SEED)
    receipt.update(seed=SEED, features=NAMES, blocks={}, checks=exact_check(model, x[vote], target[vote]),
                   protocol_sha256=b.digest(Path(__file__).with_suffix(".md")), arms={})
    for name, rows in parts.items():
        receipt["blocks"][name] = dict(n=len(rows), incomplete=int(np.isnan(x[rows]).any(axis=1).sum()),
            dates=int(frame.date.iloc[rows].nunique()), stations=int(frame.station.iloc[rows].nunique()),
            first=str(frame.date.iloc[rows].min().date()), last=str(frame.date.iloc[rows].max().date()))
    raw = {key: model.game(x[parts[key]]) for key in ("vote", "width")}
    stopped = {key: observed_game(r, x[parts[key]]) for key, r in raw.items()}
    orders, calibration = {}, {}
    for name, batch in (("ordinary", raw), ("observed", stopped)):
        games = {key: PredictiveGame.from_paths(r.a, r.incidence, target[parts[key]] - model.empty[0])
                 for key, r in batch.items()}
        orders[name] = {"mdi": np.argsort(model.model.feature_importances_, kind="stable"),
                        "min": games["vote"].ranked_path()[0]}
        calibration[name] = (batch["width"], games["width"])
    arms = {"full": np.arange(model.p)}
    spec = [("ordinary", "ordinary", False), ("ordinary", "ordinary", True),
            ("observed", "observed", False), ("observed", "observed", True), ("ordinary", "observed", True)]
    for ranking, guard, portfolio in spec:
        methods = ("mdi", "min") if portfolio else ("mdi",)
        cal, game = calibration[guard]
        paths = {}
        for m in methods:
            order = orders[ranking][m]
            curve = game.path_curves(order)[:, 1:] * (model.p / np.arange(1, model.p + 1))
            paths[m] = select.AcceptancePath(order, margin_path(model, cal, order), curve.max(axis=1), curve[:, -1])
        cols, trace = select.select_width(paths, game.null, target[width], "regression", np.std(target[fit]))
        key = f"{ranking}_{guard}_{'portfolio' if portfolio else 'mdi'}"
        arms[key] = cols
        receipt["arms"][key] = {**trace, "support": cols.tolist()}
    receipt["arms"]["full"] = dict(support=arms["full"].tolist(), width=model.p)
    results = frame.iloc[test][["date", "station"]].copy()
    results["missing"] = np.isnan(x[test]).sum(axis=1)
    for name, cols in arms.items():
        results[name + "__median"] = model.predict(x[np.ix_(test, cols)], cols)
        results[name + "__stop"] = predict_observed(model, x[test], cols)
        complete = results.missing.eq(0)
        np.testing.assert_allclose(results.loc[complete, name + "__median"], results.loc[complete, name + "__stop"])
    OUT.mkdir(exist_ok=True, parents=True)
    results.to_csv(OUT / "forecasts.csv", index=False, lineterminator="\n")
    receipt["forecast_sha256"] = b.digest(OUT / "forecasts.csv")
    (OUT / "lock.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    print("Forecasts locked", receipt["forecast_sha256"], receipt["blocks"], flush=True)


def score():
    receipt = json.loads((OUT / "lock.json").read_bytes())
    assert receipt["forecast_sha256"] == b.digest(OUT / "forecasts.csv")
    assert receipt["protocol_sha256"] == b.digest(Path(__file__).with_suffix(".md"))
    forecasts = pd.read_csv(OUT / "forecasts.csv", parse_dates=["date"])
    frame, _ = data()
    joined = forecasts.merge(frame.loc[frame.block.eq("test"), ["date", "station", "target"]], on=["date", "station"],
                             how="inner", validate="one_to_one", sort=False)
    assert len(joined) == len(forecasts) == receipt["blocks"]["test"]["n"]
    # Entire station networks remain together within a date and seven-day block.
    days = pd.date_range(joined.date.min(), joined.date.max())
    labels = days.get_indexer(joined.date)
    rng = np.random.default_rng(71201)
    starts = rng.integers(len(days), size=(2000, (len(days) + 6) // 7))
    draws = ((starts[:, :, None] + np.arange(7)) % len(days)).reshape(2000, -1)[:, :len(days)]
    scores, paired = [], []
    for subgroup, mask in (("all", np.ones(len(joined), bool)), ("incomplete", joined.missing.gt(0)),
                           ("complete", joined.missing.eq(0))):
        subset = joined.loc[mask]
        dates = labels[mask]
        day_counts = np.bincount(dates, minlength=len(days))
        n_draw = day_counts[draws].sum(axis=1)
        ok = n_draw > 0
        y = subset.target.to_numpy()
        predictions = [k for k in forecasts if "__" in k]
        for name in predictions:
            pred = subset[name].to_numpy()
            arm, mode = name.split("__")
            residual = pred - y
            scores.append(dict(group=subgroup, arm=arm, inference=mode, k=receipt["arms"][arm]["width"],
                n=len(y), stations=subset.station.nunique(), dates=subset.date.nunique(),
                missing_measurements=int(subset.missing.sum()), mean_target=y.mean(), mean_pred=pred.mean(),
                mse=np.square(residual).mean(), **b.metrics(y, pred, "regression", max(y.std(), 1e-12))))
            references = ["full__" + mode] if arm != "full" else []
            if mode == "stop":
                references.append(arm + "__median")
                if arm == "observed_observed_portfolio":
                    references += ["ordinary_ordinary_portfolio__stop", "ordinary_observed_portfolio__stop",
                                   "observed_observed_mdi__stop"]
            for reference in references:
                base = subset[reference].to_numpy() - y
                for metric, delta in (("mse", residual**2 - base**2), ("mae", np.abs(residual) - np.abs(base))):
                    by_date = np.bincount(dates, weights=delta, minlength=len(days))
                    bootstrap = by_date[draws[ok]].sum(axis=1) / n_draw[ok]
                    lo, hi = np.quantile(bootstrap, [.025, .975])
                    paired.append(dict(group=subgroup, method=name, reference=reference, metric=metric,
                        delta=delta.mean(), low=lo, high=hi, usable=int(ok.sum())))
    pd.DataFrame(scores).drop(columns="scaled_mae").to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["collect", "score"])
    with threadpool_limits(1):
        (collect if parser.parse_args().action == "collect" else score)()
