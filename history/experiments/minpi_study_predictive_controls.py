"""Resample equal-budget random supports on the three fixed transfer cohorts."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.minpi_study import ROOT, file_hash, train
from experiments.minpi_study_new_cohorts import load_cohort


RESULTS = ROOT / "results/minpi-study"
PROTOCOL = ROOT / "docs/minpi-study/predictive-controls-protocol.md"
DRAWS = 256
ENTROPY = 20261116
METHODS = ("sign", "min", "largest", "f", "random", "full")


def balanced_accuracy(y, predictions):
    labels, indices, counts = np.unique(y, return_inverse=True, return_counts=True)
    weights = 1 / (len(labels) * counts[indices])
    return float(weights[y == predictions].sum())


def score_source(name, y, group, predictions):
    if name == "supercon":
        return {"pooled_r2": 1 - np.sum((y - predictions)**2) / np.sum((y - y.mean())**2)}
    unit_scores = {}
    for unit in np.unique(group):
        rows = group == unit
        if name == "gas":
            unit_scores[f"batch_{unit}"] = balanced_accuracy(y[rows], predictions[rows])
        else:
            unit_scores[f"subject_{unit}_lower"] = min(
                balanced_accuracy(y[rows], device[rows]) for device in predictions)
    return unit_scores


def source_data(name):
    if name == "device":
        window_path = ROOT / "data/revision-cohorts/mhealth-windows.npz"
        records = RESULTS / "device/records.npz"
        supports = pd.read_csv(RESULTS / "device/supports.csv").set_index("method").indices
        with np.load(window_path) as windows, np.load(records) as case:
            ids, X = windows["ids"], windows["windows"]
            rows, test = case["dev"], case["test"]
            x_fit = X[rows, case["train_device"]]
            y_fit = ids[rows, 1]
            x_test = X[test]
            labels, groups = ids[test, 1], ids[test, 0]
            keys = {"min": "minimum", "f": "F"}
            primary = {}
            for method in METHODS:
                key = keys.get(method, method)
                primary[method] = (np.fromstring(supports[key], sep=" ", dtype=int),
                                   case[f"{key}_predictions"])

        def fit(columns):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="lbfgs",
                                                                       max_iter=2000))
            model.fit(x_fit[:, columns], y_fit)
            return np.stack([model.predict(x_test[:, d, :][:, columns]) for d in (0, 1)])

        expected = np.random.default_rng(np.random.SeedSequence(20261123, spawn_key=(3,)))
        original_draw = np.sort(expected.choice(40, 4, replace=False))
        paths = [window_path, records, RESULTS / "device/supports.csv", RESULTS / "device/source.txt"]
    else:
        X, y, group, rows, test, task = load_cohort(name)
        split_file = RESULTS / f"new-cohorts/{name}-split.npz"
        with np.load(split_file) as split:
            for key, array in (("train", rows), ("test", test), ("groups", group), ("y", y)):
                np.testing.assert_array_equal(split[key], array)
        x_fit, y_fit, x_test = X[rows], y[rows], X[test]
        labels, groups = y[test], group[test]
        primary_file = RESULTS / f"new-cohorts/{name}-1129.npz"
        with np.load(primary_file) as case:
            primary = {m: (case[f"columns_{m}"], case[f"prediction_{m}"]) for m in METHODS}

        def fit(columns):
            return train("ridge", x_fit[:, columns], y_fit, x_test[:, columns], task, 1129)[0]

        original_draw = np.sort(np.random.default_rng(1129).permutation(X.shape[1])[
            :len(primary["min"][0])])
        paths = [ROOT / f"data/revision-cohorts/{name}.zip", split_file, primary_file]
    np.testing.assert_array_equal(primary["random"][0], original_draw)
    np.testing.assert_array_equal(primary["min"][0], primary["largest"][0])
    np.testing.assert_array_equal(primary["min"][1], primary["largest"][1])
    np.testing.assert_allclose(fit(original_draw), primary["random"][1], rtol=1e-10, atol=1e-9)
    return fit, labels, groups, primary, paths


def experiment(directory):
    primary_rows, repeats, summaries = [], [], []
    inputs = [PROTOCOL, Path(__file__),
              ROOT / "experiments/minpi_study.py", ROOT / "experiments/minpi_study_device.py",
              ROOT / "experiments/minpi_study_new_cohorts.py"]
    streams = np.random.SeedSequence(ENTROPY).spawn(3)
    for name, stream in zip(("gas", "supercon", "device"), streams, strict=True):
        fit, y, groups, primary, paths = source_data(name)
        inputs.extend(paths)
        rng = np.random.default_rng(stream)
        p = len(primary["full"][0])
        h = len(primary["min"][0])
        primary_scores = {}
        for method, (columns, predictions) in primary.items():
            scores = score_source(name, y, groups, predictions)
            quality = float(np.mean(list(scores.values())))
            primary_scores[method] = quality
            primary_rows.append(dict(source=name, method=method, quality=quality,
                                     support=" ".join(map(str, columns)), **scores))
        qualities, draw_columns, draw_predictions = [], [], []
        for draw in range(DRAWS):
            columns = np.sort(rng.choice(p, h, replace=False))
            predictions = fit(columns)
            scores = score_source(name, y, groups, predictions)
            quality = float(np.mean(list(scores.values())))
            qualities.append(quality)
            repeats.append(dict(source=name, draw=draw, quality=quality,
                                support=" ".join(map(str, columns)), **scores))
            draw_columns.append(columns)
            draw_predictions.append(predictions)
        np.savez_compressed(directory / f"{name}-draws.npz", columns=draw_columns,
                            predictions=draw_predictions, test_y=y, test_groups=groups)
        values = np.array(qualities)
        pivot = primary_scores["min"]
        q05, median, q95 = np.quantile(values, [.05, .5, .95])
        summaries.append(dict(source=name, p=p, h=h, draws=DRAWS, minpi=pivot,
                              original_random=primary_scores["random"], mean=values.mean(),
                              median=median, q05=q05, q95=q95, above_minpi=(values > pivot).sum(),
                              equal_minpi=(values == pivot).sum()))
        print(name, summaries[-1], flush=True)
    pd.DataFrame(primary_rows).to_csv(directory / "primary.csv", index=False)
    pd.DataFrame(repeats).to_csv(directory / "draws.csv", index=False)
    pd.DataFrame(summaries).to_csv(directory / "summary.csv", index=False)
    manifest = {"draws": DRAWS, "seed_entropy": ENTROPY,
                "input_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in inputs}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "predictive-controls")
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1), TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "run"
        staged.mkdir()
        experiment(staged)
        staged.rename(output)
