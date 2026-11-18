"""Characterize transfer using the frozen training comparisons and labelled fitting rows."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from experiments.audit_minpi_study_mechanism import exact_row_weight
from experiments.minpi_study_calibration import individuals
from experiments.minpi_study_device import DATA
from experiments.minpi_study_mechanism import ROOT, WORLDS


RESULTS = ROOT / "results/minpi-study"


def training_ballot(votes, tiers, h, device):
    """Reconstruct G16 by integrating the exact rowwise pivots for its two focal signs."""
    p = votes.shape[1]
    pos = (votes > 0).sum(axis=1)
    neg = (votes < 0).sum(axis=1)
    zero = p - pos - neg
    counts = list(zip(pos.tolist(), neg.tolist(), zero.tolist()))
    weights = {}
    for sign, sign_counts in [(1, pos), (-1, neg)]:
        weights[sign] = np.array([exact_row_weight(a, b, c, 16, sign) if n else 0.
                                  for (a, b, c), n in zip(counts, sign_counts)])
    contributions = np.where(votes > 0, weights[1][:, None], 0.)
    contributions -= np.where(votes < 0, weights[-1][:, None], 0.)
    np.testing.assert_allclose(contributions.mean(axis=0), tiers[-1], atol=4e-15)
    np.testing.assert_allclose(votes.mean(axis=0), tiers[0], atol=4e-15)
    mass = np.abs(contributions).mean(axis=1)
    np.testing.assert_allclose(mass, (pos * weights[1] + neg * weights[-1]) / p)
    partitions = [("all", np.ones(len(votes), dtype=bool))]
    if device is not None:
        partitions.extend([("same", device == 0), ("cross", device == 1)])
    rows = []
    for name, subset in partitions:
        n_active = (pos + neg)[subset]
        has_vote = n_active > 0
        plus_share = pos[subset][has_vote] / n_active[has_vote]
        rows.append(dict(stratum=name, rows=int(subset.sum()), nonempty_rows=int(has_vote.sum()),
                         abstention=float(zero[subset].mean() / p),
                         positive_share=float(plus_share.mean()),
                         absolute_imbalance=float(np.abs(2 * plus_share - 1).mean()),
                         mean_absolute_pivot=float(mass[subset].mean())))
    minima = tiers.min(axis=0)
    largest = np.argsort(-tiers[-1], kind="stable")[:h]
    minimum = np.argsort(-minima, kind="stable")[:h]
    outside = np.setdiff1d(np.arange(p), largest)
    # The budget-line margin is negative exactly on a strict invasion of the anchor support.
    cutoff = float(minima[largest].min() - minima[outside].max())
    return rows, dict(width=p, budget=h, replaced=h-len(np.intersect1d(largest, minimum)),
                      worst_anchor_minus_best_outsider=cutoff)


def alignment(X, y, device):
    """Compare labelled class means of the same feature across the two devices."""
    labels = np.unique(y)
    means = []
    scales = []
    for b in [0, 1]:
        samples = [X[(y == label) & (device == b)] for label in labels]
        if any(not len(block) for block in samples):
            raise ValueError("Both devices must represent every class")
        means.append(np.array([block.mean(axis=0) for block in samples]))
        overall = means[-1].mean(axis=0)
        scales.append(np.mean([((block-overall)**2).mean(axis=0) for block in samples], axis=0))
    centered = [m-m.mean(axis=0) for m in means]
    pooled_var = np.mean(scales, axis=0)
    cov = (centered[0] * centered[1]).mean(axis=0)
    weights = np.divide(cov, pooled_var, out=np.zeros_like(cov), where=pooled_var > 0)
    return weights


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    strata, supports, profiles, coordinates = [], [], [], []

    def record_alignment(dataset, run, omitted, X, y, device, tiers, h):
        weights = alignment(X, y, device)
        largest = np.argsort(-tiers[-1], kind="stable")[:h]
        minimum = np.argsort(-tiers.min(axis=0), kind="stable")[:h]
        profiles.append(dict(dataset=dataset, run=run, omitted=omitted,
                             negative_mass=np.maximum(-weights, 0).sum()/np.abs(weights).sum(),
                             negative_coordinates=int((weights < 0).sum()), coordinates=len(weights),
                             negative_largest=int((weights[largest] < 0).sum()),
                             negative_minimum=int((weights[minimum] < 0).sum()), budget=h))
        if not omitted:
            for j, value in enumerate(weights):
                coordinates.append(dict(dataset=dataset, run=run, feature_index=j,
                                        normalized_covariance=value, largest=j in largest, minimum=j in minimum))

    def diagnose(name, run_id, votes, tiers, h, device=None):
        rows, diagnostic = training_ballot(votes, tiers, h, device)
        strata.extend(dict(dataset=name, run=run_id, **row) for row in rows)
        supports.append(dict(dataset=name, run=run_id, **diagnostic))

    mechanism = RESULTS / "mechanism/acquisition-bias/policy-2.npz"
    with np.load(mechanism) as case:
        all_votes, all_tiers, devices = (case[k] for k in ("train_votes","train_tiers","train_instrument"))
        types, loadings = case["types"], case["loadings"]
        np.testing.assert_allclose(case["background_match"], 5/6)
        for world in range(WORLDS):
            diagnose("simulation", world, all_votes[world], all_tiers[world], 6, devices[world])
            fit_rng = np.random.default_rng(np.random.SeedSequence([20261121, world]).spawn(5)[3])
            X, y, b = individuals(fit_rng, 384, types[world], loadings[world])
            record_alignment("simulation", world, 0, X, y, b, all_tiers[world], 6)
    fair_file = mechanism.with_name("policy-0.npz")
    fair_stats = []
    with np.load(fair_file) as fair, np.load(mechanism) as biased:
        for key in ("types", "loadings", "train_instrument", "test_y", "test_instrument"):
            np.testing.assert_array_equal(fair[key], biased[key])
        np.testing.assert_allclose(fair["background_match"], .5)
        votes, tiers, devices = (fair[key] for key in ("train_votes","train_tiers","train_instrument"))
        for world in range(WORLDS):
            rows, _ = training_ballot(votes[world], tiers[world], 6, devices[world])
            fair_stats.extend(dict(world=world, **row) for row in rows)
    pd.DataFrame(fair_stats).to_csv(output / "fair-policy-ballots.csv", index=False)
    device_file = RESULTS / "device/records.npz"
    with np.load(device_file) as case, np.load(DATA / "mhealth-windows.npz") as windows:
        ids, X = windows["ids"], windows["windows"]
        dev = case["dev"]
        assert np.array_equal(np.unique(ids[dev, 0]), np.arange(1, 6))
        b = case["train_device"]
        diagnose("MHEALTH", 0, case["train_votes"], case["rates"], 4, case["train_comparisons"][:, 3])
        for omitted in range(6):
            ix = ids[dev, 0] != omitted
            record_alignment("MHEALTH", 0, omitted, X[dev[ix], b[ix]], ids[dev[ix], 1], b[ix], case["rates"], 4)
        pairs = []
        test = case["test"]
        for subject in range(6,11):
            mask = ids[test, 0] == subject
            truth = ids[test[mask], 1]
            scores = {}
            for m in ("sign", "largest", "minimum"):
                scores[m] = min(balanced_accuracy_score(truth, pred[mask])
                                for pred in case[f"{m}_predictions"])
            pairs.append(dict(subject=subject, **scores, delta_sign=scores["minimum"]-scores["sign"],
                              delta_largest=scores["minimum"]-scores["largest"]))
    paired = pd.DataFrame(pairs)
    old = pd.read_csv(RESULTS / "device/by-subject.csv").pivot(index="subject", columns="method",
                                                               values="lower_ba")
    for m in ("sign", "largest", "minimum"):
        np.testing.assert_allclose(paired[m], old.loc[paired.subject, m])
    paired.to_csv(output / "test-people.csv", index=False)
    cohorts = RESULTS / "new-cohorts"
    cohort_files = []
    for name, h in (("gas",13), ("supercon",9)):
        for seed in (1129,2027,3469):
            source = cohorts / f"{name}-{seed}.npz"
            cohort_files.append(source)
            with np.load(source) as case:
                diagnose(name, seed, case["train_votes"], case["train_layers"], h)
    pd.DataFrame(strata).to_csv(output / "ballots.csv", index=False)
    pd.DataFrame(supports).to_csv(output / "supports.csv", index=False)
    pd.DataFrame(profiles).to_csv(output / "class-profiles.csv", index=False)
    pd.DataFrame(coordinates).to_csv(output / "feature-covariances.csv", index=False)
    files = [Path(__file__), mechanism, fair_file, device_file, DATA/"mhealth-windows.npz",
             RESULTS/"device/by-subject.csv", *cohort_files,
             ROOT/"experiments/minpi_study_calibration.py", ROOT/"experiments/audit_minpi_study_mechanism.py"]
    (output / "inputs.sha256").write_text("".join(
        f"{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n" for p in files),
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
