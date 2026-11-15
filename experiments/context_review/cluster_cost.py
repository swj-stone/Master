"""Paired task/fold bootstrap and finite-cost decisions on the archived Table 2 proposals."""
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "configs/context_review/cluster_cost.json"
OUTPUT = ROOT / "results/context_review/cluster-cost"
EPS = 1e-12


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_counts(spec):
    source = ROOT / spec["input"]
    if digest(source) != spec["sha256"]:
        raise ValueError("The archived proposal table differs from the locked input.")
    frame = pd.read_csv(source)
    keys = ["dataset", "fold", "subset"]
    assert not frame.duplicated(keys + ["cohort"]).any()
    assert frame.groupby("cohort").size().reindex(spec["cohorts"]).tolist() == spec["cohort_sizes"]
    overlap = frame.loc[frame.duplicated(keys, keep=False)]
    assert len(overlap) == 2 * spec["shared_supports"]
    assert overlap.groupby(keys)[["k", "anchor_k", "fail", *spec["rules"]]].nunique().eq(1).all().all()
    assert all(frame[k].dtype == bool for k in ["fail", *spec["rules"]])
    assert (frame.k.gt(0) & frame.k.lt(frame.anchor_k)).all()
    frame["no_veto"] = False
    frame["veto_all"] = True
    frame["drop_quarter"] = 4 * (frame.anchor_k - frame.k) >= frame.anchor_k
    datasets = sorted(frame.dataset.unique())
    n_tasks, n_folds = spec["tasks"], spec["folds"]
    cohorts, rules = spec["cohorts"], spec["rules"] + spec["extra_rules"]
    assert len(datasets) == n_tasks and sorted(frame.fold.unique()) == list(range(n_folds))
    counts = np.zeros((n_tasks, n_folds, len(cohorts), len(rules), 4), dtype=int)
    audit_rows, nesting = [], {}
    for c, cohort in enumerate(cohorts):
        sub = frame.loc[frame.cohort == cohort]
        for rule in rules:
            nested_above = (sub.separating >= sub[rule]).all()
            nested_below = (sub.separating <= sub[rule]).all()
            nesting[cohort, rule] = 1 if nested_above else -1 if nested_below else 0
        for (dataset, fold), part in sub.groupby(["dataset", "fold"]):
            veto = part[rules].to_numpy()
            event = part[["fail"]].to_numpy()
            tp = (veto & event).sum(axis=0)
            fp = (veto & ~event).sum(axis=0)
            n_event, n = event.sum(), len(part)
            counts[datasets.index(dataset), fold, c] = np.column_stack(
                (tp, fp, np.full(len(rules), n_event), np.full(len(rules), n)))
            for i, rule in enumerate(rules):
                audit_rows.append(dict(dataset=dataset, fold=fold, cohort=cohort, rule=rule,
                                       n=n, event=n_event, nonevent=n-n_event,
                                       tp=tp[i], fp=fp[i], fn=n_event-tp[i], tn=n-n_event-fp[i]))
    assert (counts[..., 3] > 0).all(), "The locked equal-fold target requires proposals in each fold."
    return counts, pd.DataFrame(audit_rows), nesting


def bootstrap(counts, spec):
    """Sufficient counts; each sampled task occurrence receives its own fold multiplicities."""
    n_tasks, n_folds = counts.shape[:2]
    n_rep, batch = spec["replicates"], spec["batch"]
    rng = np.random.default_rng(spec["seed"])
    by_fold = {
        "pooled": counts,
        "equal_task_fold": counts / counts[..., 3:4],
    }
    runs = {}
    for weighting, unit in by_fold.items():
        point = unit.sum(axis=(0, 1))
        for scheme in ("task_fold", "task_only"):
            runs[weighting, scheme] = (point, np.empty((n_rep, *counts.shape[2:]), dtype=float))
    for start in range(0, n_rep, batch):
        stop = min(start + batch, n_rep)
        size = stop - start
        tasks = rng.integers(n_tasks, size=(size, n_tasks))
        folds = rng.integers(n_folds, size=(size, n_tasks, n_folds))
        sampled = n_folds * tasks[:, :, None] + folds
        hierarchical = np.zeros((size, n_tasks * n_folds), dtype=int)
        np.add.at(hierarchical, (np.arange(size)[:, None, None], sampled), 1)
        task_only = np.zeros((size, n_tasks), dtype=int)
        np.add.at(task_only, (np.arange(size)[:, None], tasks), 1)
        assert np.all(hierarchical.sum(axis=1) == n_tasks*n_folds)
        assert np.array_equal(hierarchical.reshape(size, n_tasks, n_folds).sum(axis=2), task_only*n_folds)
        for weighting, unit in by_fold.items():
            for scheme, weights in (("task_fold", hierarchical), ("task_only", task_only.repeat(n_folds, axis=1))):
                target = runs[weighting, scheme][1][start:stop]
                target[:] = (weights @ unit.reshape(n_tasks*n_folds, -1)).reshape(target.shape)
    return runs


def divide(x, y):
    return np.divide(x, y, out=np.full(np.broadcast_shapes(np.shape(x), np.shape(y)), np.nan), where=y != 0)


def fields(cells):
    tp, fp, pos, total = cells.T
    return dict(tpr=divide(tp, pos), fpr=divide(fp, total-pos), tp_mass=tp/total, fp_mass=fp/total,
                fn_mass=(pos-tp)/total, prevalence=pos/total)


def interval(estimates):
    point, draws = estimates[0], estimates[1:]
    valid = draws[np.isfinite(draws)]
    lo, hi = np.quantile(valid, [.025, .975], method="inverted_cdf") if len(valid) else (np.nan, np.nan)
    return dict(estimate=point, lo95=lo, hi95=hi, valid_draws=len(valid))


def clean_zero(x):
    return np.where(np.abs(x) < EPS, 0., x)


def preference(a, b):
    """Characterize a - r*b < 0 for finite r>=0. Report equality separately."""
    if b == 0:
        if a == 0:
            return "never", np.nan, "every_ratio"
        return ("every_ratio" if a < 0 else "never"), np.nan, "none"
    root = a/b
    if root < 0:
        return ("every_ratio" if b > 0 else "never"), np.nan, "none"
    if b > 0:
        return "above_crossing", root, "at_crossing"
    return ("below_crossing" if root > 0 else "never"), root, "at_crossing"


def threshold_rows(a, b, orientation):
    """Bounds for nested comparisons retain draws with zero detections or complete ties."""
    a, b = clean_zero(a), clean_zero(b)
    strict_win, crossing, ties = preference(a[0], b[0])
    A, B = a[1:], b[1:]
    record = dict(delta_fp_mass=a[0], delta_tp_mass=b[0], separating_lower_cost=strict_win,
                  empirical_crossing=crossing, equal_cost=ties, draws_b_positive=int((B > 0).sum()),
                  draws_b_negative=int((B < 0).sum()), draws_b_zero=int((B == 0).sum()),
                  draws_identical_risk=int(((A == 0) & (B == 0)).sum()),
                  draws_parallel_worse=int(((A > 0) & (B == 0)).sum()),
                  draws_parallel_better=int(((A < 0) & (B == 0)).sum()),
                  nested_orientation=orientation, crossing_lo95=np.nan, crossing_hi95=np.nan)
    if orientation:
        A, B = orientation*A, orientation*B
        assert (A >= 0).all() and (B >= 0).all()
        roots = np.divide(A, B, out=np.full(A.shape, np.inf), where=B > 0)
        lower = np.where((A == 0) & (B == 0), 0, roots)
        record["crossing_lo95"] = np.quantile(lower, .025, method="inverted_cdf")
        record["crossing_hi95"] = np.quantile(roots, .975, method="inverted_cdf")
    return record


def summarize(runs, spec, nesting):
    rates, contrasts, costs, thresholds, transported = [], [], [], [], []
    rules = spec["rules"] + spec["extra_rules"]
    pairs = [(r, "endpoints") for r in rules if r != "endpoints"]
    pairs += [("separating", r) for r in rules if r not in ("separating", "endpoints")]
    for (weighting, scheme), (point, draws) in runs.items():
        for c, cohort in enumerate(spec["cohorts"]):
            keys = dict(cohort=cohort, weighting=weighting, bootstrap=scheme)
            metrics = {}
            for i, rule in enumerate(rules):
                values = fields(np.vstack((point[c, i], draws[:, c, i])))
                metrics[rule] = values
                for name, value in values.items():
                    rates.append(dict(**keys, rule=rule, metric=name, **interval(value)))
            for first, ref in pairs:
                pair = dict(**keys, rule=first, reference=ref)
                diff = {k: metrics[first][k]-metrics[ref][k] for k in metrics[first]}
                for name in ("tpr", "fpr", "fp_mass", "fn_mass"):
                    contrasts.append(dict(**pair, metric=name, **interval(diff[name])))
                for r in spec["cost_ratios"]:
                    d = clean_zero(diff["fp_mass"] + r*diff["fn_mass"])
                    costs.append(dict(**pair, ratio=r, **interval(d),
                                      draws_strictly_lower=int((d[1:] < 0).sum()),
                                      draws_tied=int((d[1:] == 0).sum())))
                if first != "separating":
                    continue
                thresholds.append(dict(**pair, **threshold_rows(diff["fp_mass"], diff["tp_mass"], nesting[cohort, ref])))
                if scheme == "task_fold":
                    for p in spec["prevalences"]:
                        pref, boundary, equal = preference(*clean_zero([(1-p)*diff["fpr"][0], p*diff["tpr"][0]]))
                        transported.append(dict(cohort=cohort, weighting=weighting, reference=ref,
                                                prevalence=p, separating_lower_cost=pref,
                                                empirical_crossing=boundary, equal_cost=equal))
    return {"rates": rates, "paired-rates": contrasts, "paired-costs": costs,
            "thresholds": thresholds, "prevalence-sensitivity": transported}


def exact_frontier(counts, spec):
    """Intersect exact rational pairwise inequalities in r; all ties are retained."""
    rules = spec["rules"] + spec["extra_rules"]
    out = []
    for c, cohort in enumerate(spec["cohorts"]):
        for weighting in ("pooled", "equal_task_fold"):
            risk = {}
            for i, rule in enumerate(rules):
                intercept = slope = Fraction(0)
                for tp, fp, pos, n in counts[:, :, c, i].reshape(-1, 4).tolist():
                    denom = 1 if weighting == "pooled" else n
                    intercept += Fraction(fp, denom)
                    slope += Fraction(pos-tp, denom)
                risk[rule] = (intercept, slope)
            for family, candidates in (("published_plus_constants", rules[:-1]), ("including_size", rules)):
                for rule in candidates:
                    intercept, slope = risk[rule]
                    lo, hi = Fraction(0), None
                    feasible = True
                    for other in candidates:
                        a, b = intercept-risk[other][0], slope-risk[other][1]
                        if b == 0 and a > 0:
                            feasible = False
                            break
                        if b > 0:
                            hi = min(hi, -a/b) if hi is not None else -a/b
                        elif b < 0:
                            lo = max(lo, -a/b)
                    if feasible and (hi is None or lo <= hi):
                        out.append(dict(cohort=cohort, weighting=weighting, family=family, rule=rule,
                                        from_ratio=float(lo), through_ratio=float(hi) if hi is not None else np.inf,
                                        exact_from=str(lo), exact_through=str(hi) if hi is not None else "inf"))
    return out


def main():
    spec = json.loads(DESIGN.read_bytes())
    counts, fold_counts, nesting = load_counts(spec)
    runs = bootstrap(counts, spec)
    tables = summarize(runs, spec, nesting)
    tables["frontier"] = exact_frontier(counts, spec)
    tables["fold-counts"] = fold_counts
    OUTPUT.mkdir(exist_ok=True, parents=True)
    receipts = {}
    for name, rows in tables.items():
        path = OUTPUT / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        receipts[path.name] = digest(path)
    manifest = dict(input_sha256=spec["sha256"], design_sha256=digest(DESIGN),
                    script_sha256=digest(Path(__file__)), numpy=np.__version__, pandas=pd.__version__,
                    seed=spec["seed"], replicates=spec["replicates"], output_sha256=receipts)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tables)} tables for {spec['replicates']:,} paired task/fold and task-only draws.")


if __name__ == "__main__":
    main()
