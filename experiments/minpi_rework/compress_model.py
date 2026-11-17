"""Train a numeric XGBoost model and deploy its MinPI-selected tree restriction."""
import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np


@dataclass
class CompactForest:
    """A pruned ensemble whose split indices refer only to the selected inputs."""
    names: tuple
    support: np.ndarray
    classes: np.ndarray
    offset: np.ndarray
    roots: np.ndarray
    heads: np.ndarray
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    default_left: np.ndarray
    value: np.ndarray

    @classmethod
    def from_forest(cls, forest, support, names, classes):
        positions = {int(col): pos for pos, col in enumerate(support)}
        roots, left, right, feature, threshold, default_left, value = [], [], [], [], [], [], []

        def visit(tree, node):
            i = len(left)
            left.append(-1)
            right.append(-1)
            feature.append(-1)
            threshold.append(0.)
            default_left.append(False)
            value.append(forest.value[tree, node])
            col = forest.feature[tree, node]
            if forest.left[tree, node] >= 0 and col in positions:
                feature[i] = positions[col]
                threshold[i] = forest.cut[tree, node]
                default_left[i] = forest.nan_left[tree, node]
                left[i] = visit(tree, forest.left[tree, node])
                right[i] = visit(tree, forest.right[tree, node])
            return i

        for tree in range(len(forest.heads)):
            roots.append(visit(tree, 0))
        return cls(tuple(names[j] for j in support), np.array(support, dtype=np.int64),
                   np.asarray(classes), forest.offset.copy(), np.asarray(roots, np.int32),
                   np.asarray(forest.heads, np.int32), np.asarray(left, np.int32),
                   np.asarray(right, np.int32), np.asarray(feature, np.int32),
                   np.asarray(threshold, np.float32), np.asarray(default_left, bool), np.asarray(value, float))

    def predict_margin(self, x):
        """Rows x retained columns in names order, yielding one margin per output group."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(self.names):
            raise ValueError(f"Supply exactly {len(self.names)} retained columns in model.names order")
        if np.isinf(x).any():
            raise ValueError("Infinite predictor; use NaN for missing measurements")
        sums = np.zeros((len(x), len(self.offset)))
        for root, head in zip(self.roots, self.heads):
            nodes = np.full(len(x), root, dtype=np.int32)
            rows = np.arange(len(x))
            while len(rows):
                rows = rows[self.feature[nodes[rows]] >= 0]
                if not len(rows):
                    break
                parent = nodes[rows]
                v = x[rows, self.feature[parent]]
                go_left = np.where(np.isnan(v), self.default_left[parent], v < self.threshold[parent])
                nodes[rows] = np.where(go_left, self.left[parent], self.right[parent])
            sums[:, head] += self.value[nodes]
        return self.offset + sums

    def predict_proba(self, x):
        if not len(self.classes):
            raise ValueError("Probabilities require classification")
        logits = self.predict_margin(x)
        if logits.shape[1] == 1:
            prob = np.exp(-np.logaddexp(0., -logits[:, 0]))
            return np.column_stack((1. - prob, prob))
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, x):
        if not len(self.classes):
            return self.predict_margin(x)[:, 0]
        return self.classes[self.predict_proba(x).argmax(axis=1)]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        # A reader sees either the old complete archive or the new complete one.
        with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with tmp_path.open("wb") as target:
                np.savez_compressed(target, version=1, **{**vars(self), "names": np.asarray(self.names, dtype=str)})
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["version"]) != 1:
                raise ValueError("Unsupported compressed forest version")
            model = cls(**{field: saved[field] for field in cls.__dataclass_fields__})
        model.names = tuple(model.names.tolist())
        return model


def fit_compressor(x_train, y_train, x_calibration, *, task, feature_names=None, rho=.2, seed=42,
                   device="cpu", n_estimators=100, max_depth=4, learning_rate=.1):
    """Fit using training labels, then restrict the tree on separately supplied unlabeled rows."""
    from minppi_mofs.modeling import make_model
    from minppi_mofs.stratified_tree import greedy_path
    from experiments.minpi_rework.quota_compress_tree import extract

    if task not in {"regression", "classification"} or not np.isfinite(rho) or rho < 0:
        raise ValueError("Provide a valid task and finite nonnegative rho")
    x = np.asarray(x_train, dtype=np.float32)
    cal = np.asarray(x_calibration, dtype=np.float32)
    y = np.asarray(y_train)
    if x.ndim != 2 or cal.ndim != 2 or min(x.shape) < 1 or not len(cal) or cal.shape[1] != x.shape[1]:
        raise ValueError("Expected nonempty train and calibration matrices with identical feature schema")
    if y.shape != (len(x),) or np.isinf(x).any() or np.isinf(cal).any():
        raise ValueError("Each training row needs a label and predictors must be finite or NaN")
    p = x.shape[1]
    names = tuple(f"f{j}" for j in range(p)) if feature_names is None else tuple(feature_names)
    if len(names) != p or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != p:
        raise ValueError("Use exactly one unique, nonempty string name per input")
    if y.dtype.kind == "O" and all(isinstance(v, str) for v in y):
        y = y.astype(str)
    if y.dtype.kind not in "bifuUS" or (y.dtype.kind in "if" and not np.isfinite(y).all()):
        raise ValueError("Labels must be nonmissing numeric values or class names")
    classes = np.empty(0, dtype=int)
    problem = task
    if task == "classification":
        classes, y = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("Classification training requires at least two classes")
        problem = "binary" if len(classes) == 2 else "multiclass"
    else:
        y = y.astype(float)
        if not np.isfinite(y).all():
            raise ValueError("Regression labels must be finite")
    started = perf_counter()
    params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=1)
    teacher = make_model(task=problem, seed=seed, backend=device, model_params=params)
    teacher.fit(x, y)
    forest = extract(teacher, cal[:1])
    game = forest.cache(cal)
    gains = teacher.get_booster().get_score(importance_type="gain")
    ranking = sorted(range(p), key=lambda j: (gains.get(f"f{j}", 0.), j), reverse=True)
    cols, info = greedy_path(game.a, game.incidence, ranking, rho=rho)
    model = CompactForest.from_forest(forest, cols, names, classes)
    retained = model.predict_margin(cal[:, cols])
    np.testing.assert_allclose(retained, forest.predict(cal[:, cols], cols), atol=1e-10, rtol=1e-10)
    full = forest.predict(cal, np.arange(p))
    err = float(np.linalg.norm(full - retained))
    scale = float(np.linalg.norm(full - forest.empty))
    if err > rho * scale + 1e-9 * max(1., scale):
        raise ArithmeticError("Serialized tree restriction exceeds its calibration fidelity budget")
    return model, dict(p=p, k=len(cols), retained_names=model.names, rho=rho, calibration_error_norm=err,
                       calibration_reference_norm=scale, calibration_rows=len(cal), nodes=len(model.left),
                       fit_compression_seconds=perf_counter()-started, **info)


def _header(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        names = next(csv.reader(source), [])
    if not names or not all(names) or len(set(names)) != len(names):
        raise ValueError(f"Empty or duplicate CSV column names in {path}")
    return names


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    fit = actions.add_parser("fit", help="Learn and export the restricted ensemble")
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--calibration", type=Path, required=True)
    fit.add_argument("--target", required=True)
    fit.add_argument("--row-id", help="Optional subject or row ID used to verify disjoint sets")
    fit.add_argument("--task", choices=["regression", "classification"], required=True)
    fit.add_argument("--rho", type=float, default=.2)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    fit.add_argument("--model", type=Path, required=True)
    predict = actions.add_parser("predict", help="Use a saved archive on the retained columns")
    predict.add_argument("--model", type=Path, required=True)
    data = predict.add_mutually_exclusive_group(required=True)
    data.add_argument("--input", type=Path)
    data.add_argument("--rows", type=int, help="Row count for a zero-column model")
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "fit":
        if args.train.samefile(args.calibration):
            parser.error("Train and calibration need separate rows in distinct files")
        if args.model.resolve() in {args.train.resolve(), args.calibration.resolve()}:
            parser.error("Choose a model output path distinct from the input files")
        names = _header(args.train)
        _header(args.calibration)
        if args.target not in names or args.target == args.row_id:
            parser.error("Target must be a column distinct from the row ID")
        predictors = [name for name in names if name not in {args.target, args.row_id}]
        df = pd.read_csv(args.train, dtype={args.row_id: str} if args.row_id else None)
        columns = predictors + ([args.row_id] if args.row_id else [])
        cal = pd.read_csv(args.calibration, usecols=columns, dtype={args.row_id: str} if args.row_id else None)
        if args.row_id:
            a, b = df[args.row_id], cal[args.row_id]
            if a.isna().any() or b.isna().any() or set(a).intersection(b):
                parser.error("Row IDs must be present and disjoint between the two files")
        model, report = fit_compressor(df[predictors], df[args.target], cal[predictors], task=args.task,
                                       feature_names=predictors, rho=args.rho, seed=args.seed, device=args.device)
        tick = perf_counter()
        model.save(args.model)
        print(json.dumps({**report, "serialization_seconds": perf_counter() - tick}, ensure_ascii=False, indent=2))
        return
    if args.output.resolve() in {args.model.resolve(), args.input.resolve() if args.input else None}:
        parser.error("Choose a prediction output path distinct from the input files")
    model = CompactForest.load(args.model)
    if args.rows is not None:
        if model.names or args.rows < 0:
            parser.error("--rows needs a zero-width model and a nonnegative row count")
        x = np.empty((args.rows, 0))
    else:
        header = _header(args.input)
        wanted = list(model.names)
        frame = pd.read_csv(args.input, usecols=wanted or [header[0]])
        x = frame[wanted].to_numpy()
    if len(model.classes):
        probs = model.predict_proba(x)
        result = pd.DataFrame({"prediction": model.classes[probs.argmax(axis=1)]})
        for j, label in enumerate(model.classes):
            result[f"probability_{label}"] = probs[:, j]
    else:
        result = pd.DataFrame({"prediction": model.predict(x)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
