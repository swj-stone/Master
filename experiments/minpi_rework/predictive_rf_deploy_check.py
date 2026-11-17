"""Check serialization, class mapping, group availability, and the signed selector."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from experiments.minpi_rework import predictive_guard as guard
from experiments.minpi_rework import predictive_rf_deploy as cli
from minppi_mofs.predictive_rf_artifact import RestrictedRF


ARM = "mdi_plus_min_min"


def reference(x1, y1, x2, y2, x3, y3, task, seed):
    x, y = np.vstack((x1, x2, x3)), np.concatenate((y1, y2, y3))
    n1, n2 = len(y1), len(y2)
    idx = dict(fit=np.arange(n1), vote=np.arange(n1, n1+n2), width=np.arange(n1+n2, len(y)), test=np.array([0]))
    forest, records, _ = guard.select(x, y, task, idx, seed)
    return forest, records[ARM]


def examples(directory, native=True):
    rng = np.random.default_rng(111)
    names = ["sensor," + str(i) for i in range(7)]
    x = rng.normal(size=(270, len(names)))
    y = x[:, 0] + .7*x[:, 1]
    x[::8, 0] = np.nan
    labels = np.array(["001", "NA", "z"])
    for task in ("classification", "regression"):
        target = labels[np.digitize(y, [-.45, .45])] if task == "classification" else y
        parts = []
        for i, rows in enumerate((slice(0, 130), slice(130, 190), slice(190, 250))):
            frame = pd.DataFrame(x[rows], columns=names)
            frame["target"] = target[rows]
            if i == 1:
                frame = frame.loc[:, list(reversed(frame.columns))]
            path = directory / f"{task}{i}.csv"
            frame.to_csv(path, index=False)
            parts.append(path)
        schema, samples = cli.from_csv(parts, "target")
        mapped = []
        for xx, yy in samples:
            mapped.extend([xx, np.searchsorted(labels, yy) if task == "classification" else yy.astype(float)])
        forest, orig = reference(*mapped, task, 42)
        cols = np.array(orig["support"], dtype=int)
        expected = RestrictedRF.from_forest(forest, cols, names, labels if task == "classification" else [], orig["winner"])
        file = directory / f"{task}.npz"
        if native:
            cli.main(["fit", "--fit", str(parts[0]), "--vote", str(parts[1]), "--accept", str(parts[2]),
                      "--task", task, "--target", "target", "--model", str(file)])
        else:
            expected.save(file)
        model = RestrictedRF.load(file)
        assert model.metadata == expected.metadata and schema == names
        test = x[250:].copy()
        reduced = test[:, cols]
        np.testing.assert_allclose(model.predict(reduced), forest.predict(reduced, cols), atol=1e-14, rtol=0)
        # Each unavailable feature suppresses the entire split, including every output class.
        avail = rng.uniform(size=reduced.shape) < .55
        reference_rows = [forest.predict(test[i:i+1, cols[mask]], cols[mask])[0] for i, mask in enumerate(avail)]
        np.testing.assert_allclose(model.predict(reduced, avail), reference_rows, atol=1e-14, rtol=0)
        # Add a declared but unreachable input before every actual index.
        from dataclasses import replace
        extended = replace(model, metadata=dict(model.metadata, features=["unused", *model.names]),
                           medians=np.r_[0., model.medians],
                           feature=np.where(model.left >= 0, model.feature + 1, model.feature))
        compact = extended.reachable_schema()
        used = [model.names.index(name) for name in compact.names]
        assert "unused" not in compact.names and compact.reachable_schema() is compact
        np.testing.assert_array_equal(compact.predict(reduced[:, used], avail[:, used]), model.predict(reduced, avail))
        frame = pd.DataFrame(reduced, columns=model.names).iloc[:, ::-1]
        flags = pd.DataFrame(avail.astype(int), columns=model.names)
        if len(cols):
            flags[model.names[0]] = 0
            frame = frame.drop(columns=[model.names[0]])
        frame["id"] = np.arange(len(test))
        frame.to_csv(directory/"inputs.csv", index=False)
        flags.to_csv(directory/"mask.csv", index=False)
        cli.main(["predict", "--model", str(file), "--input", str(directory/"inputs.csv"), "--availability",
                  str(directory/"mask.csv"), "--output", str(directory/"answers.csv")])
        answers = cli.read_table(directory/"answers.csv")
        flags = flags.to_numpy(bool)
        pred = model.predict(reduced, flags)
        if task == "classification":
            assert list(answers) == ["prediction", "probability_001", "probability_NA", "probability_z"]
            np.testing.assert_array_equal(answers.prediction, labels[pred.argmax(axis=1)])
            np.testing.assert_allclose(answers.iloc[:, 1:].astype(float), pred, atol=1e-15)
        else:
            np.testing.assert_allclose(answers.prediction.astype(float), pred, atol=1e-15)
        empty = RestrictedRF.from_forest(forest, [], names, model.classes, orig["winner"])
        np.testing.assert_allclose(empty.predict(test[:, :0]), forest.predict(test[:, :0], []), atol=1e-14)
        damaged = directory/"damaged.npz"
        with np.load(file, allow_pickle=False) as archive:
            buffers = dict(archive)
        buffers["left"][0] = 0
        np.savez(damaged, **buffers)
        try:
            RestrictedRF.load(damaged)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt edge was accepted")
        print(f"{task}: k={len(cols)}, winner={orig['winner']}, {len(avail)} availability patterns")


def sealed(directory, name, seed):
    from experiments.minpi_rework import validation_predictive as panel
    base = Path(__file__).resolve().parents[2] / "results/minpi_rework/predictive-confirmation" / f"{name}-{seed}"
    file = directory/"registered.npz"
    cli.main(["fit", "--dataset", name, "--seed", str(seed), "--model", str(file)])
    model = RestrictedRF.load(file)
    rec = json.loads((base/"selection.json").read_bytes())["records"][ARM]
    x, _, _, idx = panel.split(name, seed)
    assert model.names == [f"f{j}" for j in rec["support"]] and model.metadata["winner"] == rec["winner"]
    with np.load(base/"forecasts.npz", allow_pickle=False) as forecasts:
        np.testing.assert_allclose(model.predict(x[np.ix_(idx["test"], rec["support"])]), forecasts[ARM],
                                   atol=1e-14, rtol=0)
    print(f"{name}/{seed} sealed path, k={rec['k']}, {len(idx['test'])} archived forecasts reproduced")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", action="store_true", help="Also compare to three signed local split forecasts")
    parser.add_argument("--artifact-only", action="store_true", help="Exercise persistence directly with the signed selector")
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        examples(Path(temp), native=not args.artifact_only)
        if args.sealed:
            for name in ("votes", "german", "beijing"):
                sealed(Path(temp), name, 42)
