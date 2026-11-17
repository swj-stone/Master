"""Fit and use the supervised forest MinPI with three disjoint labeled samples."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from minppi_mofs.predictive_rf_artifact import RestrictedRF


def read_table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or any(not h for h in header) or len(set(header)) != len(header):
        raise ValueError(f"Empty or duplicate CSV headers in {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def numeric(frame):
    return frame.replace(["", "NA", "NaN", "nan", "null", "?"], np.nan).apply(
        pd.to_numeric, errors="raise").to_numpy(dtype=float)


def from_csv(paths, target):
    if len(set(Path(path).resolve() for path in paths)) != len(paths):
        raise ValueError("Supply three separate fitting, voting and acceptance files")
    frames = [read_table(path) for path in paths]
    if target not in frames[0] or any(set(f.columns) != set(frames[0].columns) for f in frames):
        raise ValueError("The three CSVs must contain the same feature names and the target")
    names = [c for c in frames[0] if c != target]
    return names, [(numeric(frame[names]), frame[target].to_numpy(dtype=str)) for frame in frames]


def fit_parts(parts, names, task, seed, selector=None, *, units=None):
    """Encode training labels once; the selector receives fit, vote, and width samples only."""
    if selector is None:
        from experiments.minpi_rework.predictive_selection import select_forest
        selector = select_forest
    if task not in ("classification", "regression") or not names or len(parts) != 3:
        raise ValueError("Provide a classification or regression task, features and three labeled samples")
    classes = np.unique(parts[0][1]) if task == "classification" else []
    labels = [str(c) for c in classes]
    if task == "classification" and (len(classes) < 2 or "" in labels):
        raise ValueError("Classification needs at least two named fitting classes")
    encoded = []
    for i, (x, y) in enumerate(parts):
        if x.ndim != 2 or x.shape != (len(y), len(names)) or not len(y) or np.isinf(x).any():
            raise ValueError("Check sample sizes and the numeric feature schema")
        if task == "classification":
            if not np.isin(y, classes).all() or (i == 2 and len(np.unique(y)) != len(classes)):
                raise ValueError("New classes or missing acceptance classes invalidate the native AUC guard")
            values = np.searchsorted(classes, y)
        else:
            values = np.asarray(y, dtype=float)
            if not np.isfinite(values).all() or (i == 2 and len(values) < 2):
                raise ValueError("Regression needs finite targets and at least two acceptance rows")
        encoded.extend((x, values))
    selection_options = {}
    if units is not None:
        if len(units) != len(names) or any(not isinstance(unit, str) or not unit for unit in units):
            raise ValueError("Provide a named acquisition unit for every input column")
        unit_names = list(dict.fromkeys(units))
        unit_index = {unit: i for i, unit in enumerate(unit_names)}
        selection_options["unit_ids"] = np.array([unit_index[unit] for unit in units])
    forest, columns, trace = selector(*encoded, task, seed=seed, epsilon=.02, **selection_options)
    artifact = RestrictedRF.from_forest(forest, columns, names, labels, trace["path"] + "_min")
    if units is not None:
        artifact.metadata["acquisition_units"] = {names[i]: units[i] for i in columns}
    return artifact


def predict_table(model, frame, masks=None, *, unit_masks=None):
    names = model.names
    available = None
    if unit_masks is not None:
        groups = model.metadata.get("acquisition_units")
        if groups is None or masks is not None:
            raise ValueError("Unit availability requires a group-trained model and one mask input")
        if len(unit_masks) != len(frame) or any(groups[j] not in unit_masks for j in names):
            raise ValueError("Unit mask rows must align with inputs and include all requested units")
        masks = pd.DataFrame({j: unit_masks[groups[j]] for j in names}, index=unit_masks.index)
    if masks is not None:
        if len(masks) != len(frame) or any(j not in masks for j in names):
            raise ValueError("Availability rows must align with input, with one 0/1 column per selected feature")
        flags = numeric(masks[names])
        if not np.isin(flags, [0, 1]).all():
            raise ValueError("Availability must be 0 or 1")
        available = flags.astype(bool)
    for j, name in enumerate(names):
        if name not in frame and (available is None or available[:, j].any()):
            raise ValueError(f"Missing available feature {name}")
    x = numeric(frame.reindex(columns=names))
    scores = model.predict(x, available)
    if not model.classes:
        return pd.DataFrame({"prediction": scores})
    return pd.DataFrame({"prediction": np.asarray(model.classes)[scores.argmax(axis=1)],
                         **{"probability_" + c: scores[:, i] for i, c in enumerate(model.classes)}})


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cmd = cli.add_subparsers(dest="action", required=True)
    fit = cmd.add_parser("fit")
    source = fit.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Registered source in validation_predictive")
    source.add_argument("--fit", type=Path)
    fit.add_argument("--vote", type=Path)
    fit.add_argument("--accept", type=Path)
    fit.add_argument("--target")
    fit.add_argument("--task", choices=["classification", "regression"])
    fit.add_argument("--seed", default=42, type=int)
    fit.add_argument("--reachable-only", action="store_true",
                     help="Export only features used by a reachable split in the selected forest")
    fit.add_argument("--groups", type=Path, help="CSV with one feature,unit row per input column")
    fit.add_argument("--model", type=Path, required=True)
    infer = cmd.add_parser("predict")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    availability = infer.add_mutually_exclusive_group()
    availability.add_argument("--availability", type=Path)
    availability.add_argument("--unit-availability", type=Path,
                              help="0/1 availability CSV indexed by the named acquisition units")
    cmd.add_parser("inspect").add_argument("--model", type=Path, required=True)
    args = cli.parse_args(argv)
    if args.action == "fit":
        if args.dataset:
            from experiments.minpi_rework import validation_predictive as panel
            if any(arg is not None for arg in (args.vote, args.accept, args.target, args.task)):
                cli.error("The registered source defines all partitions, task and target")
            if args.dataset not in panel.SOURCES:
                cli.error(f"Registered sources: {', '.join(panel.SOURCES)}")
            x, y, task, idx = panel.split(args.dataset, args.seed)
            names = [f"f{j}" for j in range(x.shape[1])]
            parts = [(x[idx[key]], y[idx[key]]) for key in ("fit", "vote", "width")]
        else:
            if not (args.vote and args.accept and args.target and args.task):
                cli.error("CSV fitting requires --vote, --accept, --target and --task")
            sources = [args.fit, args.vote, args.accept]
            if args.model.resolve() in [p.resolve() for p in sources]:
                cli.error("The output artifact must differ from its inputs")
            names, parts = from_csv(sources, args.target)
            task = args.task
        units = None
        if args.groups is not None:
            if args.groups.resolve() == args.model.resolve():
                cli.error("The output artifact must differ from the group file")
            table = read_table(args.groups)
            if (list(table.columns) != ["feature", "unit"] or table.feature.duplicated().any()
                    or set(table.feature) != set(names)):
                cli.error("The group file must assign every named feature exactly once (feature,unit)")
            units = table.set_index("feature").loc[names, "unit"].tolist()
        model = fit_parts(parts, names, task, args.seed, units=units)
        if args.reachable_only:
            model = model.reachable_schema()
        model.save(args.model)
        print(json.dumps(model.metadata, ensure_ascii=False))
    elif args.action == "inspect":
        print(json.dumps(RestrictedRF.load(args.model).metadata, ensure_ascii=False, indent=2))
    else:
        inputs = [args.model, args.input] + [p for p in (args.availability, args.unit_availability) if p]
        if args.output.resolve() in [p.resolve() for p in inputs]:
            cli.error("Prediction output must differ from inputs")
        model = RestrictedRF.load(args.model)
        masks = read_table(args.availability) if args.availability else None
        unit_masks = read_table(args.unit_availability) if args.unit_availability else None
        result = predict_table(model, read_table(args.input), masks, unit_masks=unit_masks)
        result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
