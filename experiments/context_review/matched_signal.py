"""Cross-task incremental value of context dissent over continuous masked damage."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.context_review import incremental_prediction as original
from experiments.context_review import rule_sensitivity as replay
from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import DATA, OUT as PRIMARY, ROOT, config, digest


DESIGN = ROOT / "configs/context_review/matched_signal.json"
OUT = ROOT / "results/context_review/matched-signal-r2"
EVIDENCE = ROOT / "thesis-paper/evidence/matched_signal"
SOURCES = ("path", "uniform_deletions")
KEY = ["dataset", "fold", "subset"]
CONTRASTS = [("gain", "width"), ("empty", "gain"), ("endpoint", "empty"),
             ("endpoint_vote", "endpoint"), ("all_margins", "endpoint"),
             ("all_margins_vote", "all_margins")]


def features(name, fold):
    suffix = Path(name) / f"fold-{fold}"
    base, donor, votes = PRIMARY / suffix, replay.DONOR / suffix, replay.OUT / suffix
    read = original.read
    source, sel = read(base / "selection.json"), read(donor / "selection.json")
    assert sel["donor"] == digest(base / "selection.json") == read(base / "completed.json")["selection_sha256"]
    assert digest(votes / "votes.npz") == read(votes / "voting.json")["votes"]
    assert digest(donor / "selection.json") == read(votes / "voting.json")["donor"]
    table = DATA / name / f"{name}.tsv.gz"
    assert digest(table) == read(base / "completed.json")["data_sha256"]
    data = pd.read_csv(table, sep="\t").iloc[source["train"]]
    y = data.pop("target").map({c: i for i, c in enumerate(source["class_order"])}).to_numpy()
    x = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(data.to_numpy(dtype=np.float32))
    booster = fit(x, y, 20421 + fold).get_booster()
    gain = {int(key[1:]): val for key, val in booster.get_score(importance_type="gain").items()}
    mass = {int(key[1:]): val for key, val in booster.get_score(importance_type="total_gain").items()}
    order = sorted(gain, key=lambda j: (-gain[j], j)) or [0]
    assert order == source["order"]
    anchor = frozenset(sel["anchor"])
    assert set(order[:len(anchor)]) == anchor
    ranks = {feature: len(anchor) - i for i, feature in enumerate(order[:len(anchor)])}
    ext = list(dict.fromkeys(frozenset(s) for s in sel["contexts"]["separating"]))
    assert ext[0] == frozenset()
    with np.load(votes / "votes.npz") as archived:
        tol = np.array([.05 if m == "ap" else .03 for m in archived["metrics"]])
        masks = {frozenset(np.flatnonzero(s)): j for j, s in enumerate(archived["xgb_masks"])}
        scores = archived["xgb_scores"][:, :4, :]
    reference = scores[[masks[anchor | e] for e in ext]]
    results = []
    for i, proposal in enumerate(sel["supports"]):
        support = frozenset(proposal)
        dropped = anchor - support
        reduced = scores[[masks[support | e] for e in ext]]
        excess = reference - reduced - tol * abs(reference)
        margin = (excess / tol).max(axis=2)
        endpoints = margin[[0, -1]] if len(ext) > 1 else margin[:1]
        dissent = (excess > 1e-12).any(axis=2).mean(axis=1).max()
        assert dissent == 1 - np.array(sel["ballots"][i]["separating"]).mean(axis=1).min()
        fraction = len(dropped) / len(anchor)
        results.append(dict(dataset=name, fold=fold, subset=i, k=len(support), anchor_k=len(anchor),
            path=proposal in sel["cohorts"]["path"], uniform_deletions=proposal in sel["cohorts"]["uniform_deletions"],
            x=fraction, hinge25=max(0, fraction-.25), hinge50=max(0, fraction-.50), hinge75=max(0, fraction-.75),
            gain_mass=sum(mass[j] for j in dropped) / sum(mass[j] for j in anchor),
            rank_mass=sum(ranks[j] for j in dropped) / sum(ranks.values()), point=margin[0, 0],
            endpoint_mean=endpoints.mean(), endpoint_max=endpoints.max(), all_mean=margin.mean(),
            all_max=margin.max(), vote=dissent, contexts=len(ext)))
    return results


def score_models(table, settings):
    models = {}
    for name, items in settings["models"].items():
        models[name] = [c for item in items for c in models.get(item, [item])]
    outputs, coefficients = [], []
    for source in ("union", *SOURCES):
        for held, test in table.groupby("dataset"):
            train = table.loc[(table.dataset != held) & table[source]]
            assert train.dataset.nunique() == 13 and train.fail.nunique() == 2
            w = original.weights(train)
            np.testing.assert_allclose(w.sum(), 1)
            test = test.copy()
            for name, cols in models.items():
                scaler = StandardScaler().fit(train[cols], sample_weight=w)
                beta = original.fit_logistic(scaler.transform(train[cols]), train.fail.to_numpy(), w, settings["lambda"])
                # Use the same addition order on every row, including the final
                # rows of a fold. Different BLAS kernels can split exact ties.
                logits = np.full(len(test), beta[0])
                for column, slope in zip(scaler.transform(test[cols]).T, beta[1:], strict=True):
                    logits += column * slope
                test[name] = logits
                coefficients.append(dict(train_source=source, held=held, model=name, columns=cols,
                    training_tasks=sorted(train.dataset.unique()), intercept=beta[0], slopes=beta[1:].tolist(),
                    means=scaler.mean_.tolist(), scales=scaler.scale_.tolist()))
            outputs.append(test.assign(train_source=source))
    return pd.concat(outputs, ignore_index=True), coefficients


def common_support(table):
    """Range is computed from held-task proposal geometry without outcomes."""
    table = table.copy()
    for source in SOURCES:
        by_width = table.loc[table[source]].groupby(["dataset", "fold", "k"]).gain_mass.agg(["min", "max"])
        bounds = table[["dataset", "fold", "k"]].merge(by_width, how="left", on=["dataset", "fold", "k"],
                                                       validate="many_to_one")
        table[f"in_{source}"] = table.gain_mass.between(bounds["min"] - 1e-12, bounds["max"] + 1e-12)
        assert table.loc[table[source], f"in_{source}"].all()
    return table


def summarize(part, models):
    rows = []
    for (dataset, fold), group in part.groupby(["dataset", "fold"]):
        y, k = group.fail.to_numpy(), group.k.to_numpy()
        pairs = (y[:, None] == 1) & (y[None, :] == 0)
        same = pairs & (k[:, None] == k[None, :])
        for metric, n_pairs in (("log_loss", 0), ("brier", 0), ("auc", pairs.sum()), ("same_width_auc", same.sum())):
            if metric.endswith("auc") and n_pairs == 0:
                continue
            row = dict(dataset=dataset, fold=fold, metric=metric, proposals=len(y), pairs=n_pairs, failures=y.sum())
            for model in models:
                z = group[model].to_numpy()
                if metric == "log_loss":
                    value = (np.logaddexp(0, z) - y*z).mean()
                elif metric == "brier":
                    value = np.square(expit(z) - y).mean()
                else:
                    value = original.concordance(z, same if metric == "same_width_auc" else pairs)
                if metric == "same_width_auc" and model == "width":
                    assert value == .5
                row[model] = value
            rows.append(row)
    return pd.DataFrame(rows)


def paired(folds, contrasts, settings):
    averaged = folds.groupby(["metric", "dataset"])
    tasks = averaged[[*dict.fromkeys(n for c in contrasts for n in c)]].mean()
    tasks["folds"] = averaged.size()
    tasks[["proposals", "pairs", "failures"]] = averaged[["proposals", "pairs", "failures"]].sum()
    tasks = tasks.reset_index()
    rows = []
    for metric, group in tasks.groupby("metric"):
        rng = np.random.default_rng(settings["bootstrap_seed"])
        samples = rng.integers(len(group), size=(settings["bootstrap_replicates"], len(group)))
        for plus, base in contrasts:
            changes = (group[plus] - group[base]).to_numpy()
            lo, hi = np.quantile(changes[samples].mean(axis=1), [.025, .975])
            rows.append(dict(metric=metric, augmented=plus, baseline=base, tasks=len(group),
                folds=group.folds.sum(), proposals=group.proposals.sum(), pairs=group.pairs.sum(),
                failures=group.failures.sum(), augmented_score=group[plus].mean(), baseline_score=group[base].mean(),
                difference=changes.mean(), lower=lo, upper=hi, zero_tasks=np.isclose(changes, 0, atol=1e-14).sum()))
    return tasks, pd.DataFrame(rows)


def analyze(predictions, settings):
    collected = {"folds": [], "tasks": [], "summary": []}
    models = list(settings["models"])

    def record(part, train, test, sample, names, contrasts):
        if part.empty:
            return
        folds = summarize(part, names)
        tasks, summary = paired(folds, contrasts, settings)
        for kind, table in (("folds", folds), ("tasks", tasks), ("summary", summary)):
            collected[kind].append(table.assign(train_source=train, test_source=test, sample=sample))

    for source, source_rows in predictions.groupby("train_source"):
        for target in (["union"] if source == "union" else SOURCES):
            rows = source_rows[source_rows[target]]
            record(rows, source, target, "all", models, CONTRASTS)
            if source != target:
                matched = rows[rows[f"in_{source}"]]
                record(matched, source, target, "width_gain_range", models, CONTRASTS)
                record(matched[~matched[source]], source, target, "range_exclusive", models, CONTRASTS)
    shared = predictions[predictions.path & predictions.uniform_deletions]
    cols = ["endpoint", "endpoint_vote"]
    paired_shared = shared.loc[shared.train_source.eq("path")].merge(
        shared.loc[shared.train_source.eq("uniform_deletions"), KEY + cols], on=KEY, validate="one_to_one",
        suffixes=("_path", "_uniform"))
    contrasts = [(n + "_path", n + "_uniform") for n in cols]
    record(paired_shared, "path_minus_uniform", "shared", "identical_supports",
           [c for pair in contrasts for c in pair], contrasts)
    return {key: pd.concat(parts, ignore_index=True) for key, parts in collected.items()}


def main():
    settings = original.read(DESIGN)
    OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    folds = [(name, fold) for name in config()["datasets"] for fold in range(3)]
    files = [DESIGN, Path(__file__), Path(original.__file__), Path(replay.__file__),
             ROOT / "experiments/context_study/benchmark.py", replay.OUT / "protocol.json", replay.DONOR / "protocol.json"]
    for name, fold in folds:
        suffix = Path(name) / f"fold-{fold}"
        files.append(DATA / name / f"{name}.tsv.gz")
        files.extend(p / suffix / f for p, fs in [(PRIMARY, ("selection.json", "completed.json")),
            (replay.DONOR, ("completed.json", "selection.json")), (replay.OUT, ("voting.json", "votes.npz"))] for f in fs)
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    lock = OUT / "inputs.json"
    if lock.exists():
        assert original.read(lock) == hashes, "Input/design changed since the first run"
    else:
        store(lock, hashes)
    with threadpool_limits(limits=1):
        rows = [r for name, fold in folds for r in features(name, fold)]
        table = pd.DataFrame(rows).assign(union=True)
        assert len(table) == 704 and table.path.sum() == 400 and table.uniform_deletions.sum() == 379
        table = common_support(table)
        known = original.records([replay.DONOR / name / f"fold-{fold}" for name, fold in folds], SOURCES)
        table = table.merge(known.drop_duplicates(KEY)[KEY + ["fail", "v", "x"]], on=KEY,
                            suffixes=("", "_archived"), validate="one_to_one")
        np.testing.assert_array_equal(table.vote, table.v)
        np.testing.assert_array_equal(table.x, table.x_archived)
        assert len(table) == 704 and all(table.path | table.uniform_deletions)
        predictions, fitted = score_models(table, settings)
    tables = analyze(predictions, settings)
    for name, frame in {"proposals": table, "predictions": predictions, **tables}.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
        if name not in {"folds", "predictions"}:
            frame.to_csv(EVIDENCE / f"{name}.csv", index=False)
    store(OUT / "models.json", fitted)
    store(EVIDENCE / "manifest.json", {"design_sha256": digest(DESIGN), "inputs_sha256": digest(lock),
        "outputs": {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}})
    main_row = tables["summary"].query("augmented == 'endpoint_vote' and metric == 'log_loss'")
    print(main_row.to_string(index=False))


if __name__ == "__main__":
    main()
