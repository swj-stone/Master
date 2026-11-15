"""Replay the budgeted-guard study using raw tables, public ML libraries and audit utilities."""
import argparse
from itertools import combinations
import json
from math import ceil, comb
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from rounds.pmlb_baselines import DATA, ROOT, RUN, auc, compact_gain, fit, read, sha


HERE = Path(__file__).resolve().parent
STUDY = ROOT / "results/context_review/budgeted-guard"
ELIGIBLE = ROOT / "results/context_review/refit-guard/eligibility.csv"
ARCHIVE = ROOT / "data/context_study/uci-optdigits/uci80.zip"
DIGITS_SHA = "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0"
LEARNED = ROOT / "results/context_review/learned-guard"
MODELS = ROOT / "results/context_review/incremental-prediction/models.json"


def partitions(name, fold):
    seed = 20421 + fold
    if name == "optdigits":
        assert fold == 0 and sha(ARCHIVE) == DIGITS_SHA
        with ZipFile(ARCHIVE) as archive:
            raw = pd.read_csv(archive.open("optdigits.tra"), header=None)
            future = pd.read_csv(archive.open("optdigits.tes"), header=None)
        assert raw.shape == (3823, 65) and future.shape == (1797, 65)
        table = pd.concat([raw, future], ignore_index=True)
        labels = table.pop(64).to_numpy()
        dev = np.arange(len(raw))
        outer = np.arange(len(raw), len(table))
        checksum = DIGITS_SHA
    else:
        path = DATA / name / f"{name}.tsv.gz"
        checksum = sha(path)
        assert checksum == read(RUN / "sources.json")["sources"][name][path.name]["sha256"]
        table = pd.read_csv(path, sep="\t")
        labels = table.pop("target").to_numpy()
        dev, outer = list(StratifiedKFold(3, shuffle=True, random_state=20261115)
                          .split(table, labels))[fold]
    train, rank = train_test_split(dev, test_size=.25, random_state=seed, stratify=labels[dev])
    classes, counts = np.unique(labels[train], return_counts=True)
    if len(classes) == 2 and counts[0] < counts[1]:
        classes = classes[::-1]
    y = pd.Categorical(labels, categories=classes).codes
    if name != "optdigits":
        donor = read(RUN / name / f"fold-{fold}" / "selection.json")
        for field, values in (("train", train), ("rank", rank), ("test", outer), ("class_order", classes)):
            np.testing.assert_array_equal(donor[field], values)
    if np.bincount(y[rank]).min() < 2:
        return None
    prop, guard = train_test_split(rank, test_size=.5, stratify=y[rank], random_state=39823 + fold)
    np.testing.assert_array_equal(np.sort(np.r_[train, prop, guard, outer]), np.arange(len(table)))
    assert set(y[prop]) == set(y[guard]) == set(y[train])
    # Group by the exact raw predictor row, before conversion/imputation, ignoring its label.
    identity = table.groupby(list(table.columns), dropna=False, sort=False).ngroup().to_numpy()
    return (table.to_numpy(dtype=np.float32), y, train, prop, guard, dev, outer, identity, checksum)


def candidate_sets(order, width, task, fold):
    columns = sorted(order)
    prefix = tuple(sorted(order[:width]))
    if comb(len(columns), width) <= 8:
        members = list(combinations(columns, width))
    else:
        rng = np.random.default_rng([81027, task, fold, width])
        chosen = {prefix}
        while len(chosen) < 8:
            chosen.add(tuple(sorted(rng.choice(columns, size=width, replace=False).tolist())))
        members = sorted(chosen)
    shuffled = np.random.default_rng([72291, task, fold, width]).permutation(len(members))
    gain = sorted(members, key=lambda s: (sum(order.index(i) for i in s), s))
    assert gain[0] == prefix
    return members, gain, [members[i] for i in shuffled]


def panels(y, seed):
    rng = np.random.default_rng([seed, 659])
    group_sizes = np.bincount(y)
    quota = np.zeros_like(group_sizes)
    available = min(len(y), 256)
    for used, cls in enumerate(np.argsort(group_sizes, kind="stable")):
        quota[cls] = min(group_sizes[cls], available // (len(group_sizes) - used))
        available -= quota[cls]
    sample = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == cls), q, replace=False)
                                     for cls, q in enumerate(quota)]))
    weights = np.zeros((4, len(sample)))
    for cls, q in enumerate(quota):
        mask = y[sample] == cls
        weights[0, mask] = 1
        for j in range(1, 4):
            weights[j, mask] = np.bincount(rng.integers(q, size=q), minlength=q)
        weights[:, mask] *= group_sizes[cls] / (len(y) * q)
    return sample, weights


def restorations(tail):
    if len(tail) < 2:
        return [frozenset()] + ([frozenset(tail)] if tail else [])
    bits = 2
    while comb(bits, bits // 2) < len(tail):
        bits += 1
    patterns = dict(zip(tail, combinations(range(bits), bits // 2)))
    interior = [frozenset(v for v, bits_used in patterns.items() if i in bits_used)
                for i in range(bits)]
    return [frozenset(), *interior, frozenset(tail)]


def score_panels(y, prob, weights, rare):
    # Direct weighted concordance over positive-negative pairs, with half credit for ties.
    # This formula uses no sorting of prediction scores and shares no evaluation code
    # with the experiment's weighted rank-based implementation.
    aucs = []
    for c in ([1] if prob.shape[1] == 2 else range(prob.shape[1])):
        pos, neg = prob[y == c, c], prob[y != c, c]
        concordance = (pos[:, None] > neg) + .5 * (pos[:, None] == neg)
        wp, wn = weights[:, y == c], weights[:, y != c]
        aucs.append(((wp @ concordance) * wn).sum(axis=1) / (wp.sum(axis=1) * wn.sum(axis=1)))
    accuracy = (weights @ (prob.argmax(axis=1) == y)) / weights.sum(axis=1)
    values = [np.mean(aucs, axis=0), accuracy]
    if rare:
        values.append([average_precision_score(y, prob[:, 1], sample_weight=w) for w in weights])
    return np.array(values).T


def measures(y, prob, rare):
    values = [auc(y, prob), np.mean(prob.argmax(axis=1) == y)]
    if rare:
        values.append(average_precision_score(y, prob[:, 1]))
    return np.array(values)


def accepted(candidate, reference):
    tolerances = np.array([.03, .03, .05][:reference.shape[-1]])
    return np.all(candidate + 1e-12 >= reference * (1 - tolerances), axis=-1)


def screen(model, train, proposal, labels, seed, anchor, tail, pools, rare):
    ids, weights = panels(labels, seed)
    bg_ids = np.random.default_rng(seed).choice(len(train), size=min(512, len(train)), replace=False)
    background = np.median(train[bg_ids], axis=0)
    x = proposal[ids]
    family = restorations(tail)
    lookup = {}

    def point(cols):
        if cols not in lookup:
            masked = np.broadcast_to(background, x.shape).copy()
            live = sorted(cols)
            masked[:, live] = x[:, live]
            lookup[cols] = score_panels(labels[ids], model.predict_proba(masked), weights, rare)
        return lookup[cols]

    # Compute *all* restorations, then reconstruct the cold-start short circuit separately.
    reference = [point(frozenset(anchor) | ext) for ext in family]
    approvals, visited = {}, {}
    for supports, _, _ in pools.values():
        for cols in supports:
            after = [point(frozenset(cols) | ext) for ext in family]
            sequence = [int(accepted(a, r).sum()) for a, r in zip(after, reference, strict=True)]
            approvals[cols] = min(sequence)
            count = sequence.index(0) + 1 if 0 in sequence else len(sequence)
            visited[cols] = {base | e for e in family[:count] for base in (frozenset(anchor), frozenset(cols))}
    return approvals, visited


def refits(x, y, train, target, supports, seed, path, guard=False):
    transformer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    a, b = transformer.transform(x[train]), transformer.transform(x[target])
    pred = {}
    with np.load(path) as stored:
        assert len(stored.files) == len(supports) + int(guard)
        if guard:
            np.testing.assert_array_equal(stored["y"], y[target])
        for i, cols in enumerate(sorted(supports)):
            p = fit(a[:, cols], y[train], seed).predict_proba(b[:, cols])
            np.testing.assert_array_equal(p, stored[f"s{i}"])
            pred[cols] = p
    return pred


def audit_fold(name, fold, task, data, check_learned):
    x, y, train, prop, guard, dev, outer, identity, source_sha = data
    folder = STUDY / name / f"fold-{fold}"
    listed = read(folder / "shortlists.json")
    sel = read(folder / "selection.json")
    done = read(folder / "completed.json")
    assert sha(folder / "shortlists.json") == sel["shortlists"]
    assert sha(folder / "guard.npz") == sel["guard_sha"]
    assert sha(folder / "selection.json") == done["selection"]
    assert sha(folder / "predictions.npz") == done["predictions"]
    assert listed["source"] == source_sha and listed["protocol"] == sha(STUDY / "protocol.json")
    for field, values in (("train", train), ("proposal", prop), ("guard", guard), ("dev", dev)):
        np.testing.assert_array_equal(listed[field], values)
    seed = 20421 + fold
    transform = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    a, b = transform.transform(x[train]), transform.transform(x[prop])
    model = fit(a, y[train], seed)
    order, k = compact_gain(model, a, b, y[prop], seed)
    assert order == listed["order"] and seed == listed["seed"]
    anchor = tuple(sorted(order[:k]))
    assert list(anchor) == listed["anchor"]
    if name != "optdigits":
        old = read(ELIGIBLE.parent / name / f"fold-{fold}" / "proposals.json")
        assert order == old["order"] and list(anchor) == old["candidates"]["anchor"]
        np.testing.assert_array_equal(old["test"], outer)
    rare = model.n_classes_ == 2 and np.bincount(y[prop]).min() <= .05 * len(prop)
    assert bool(rare) == sel["rare"]
    widths = sorted({ceil(k*j/4) for j in (1, 2, 3)} - {k})
    pools = {w: candidate_sets(order[:k], w, task, fold) for w in widths}
    assert {str(w): [[list(s) for s in seq] for seq in pool] for w, pool in pools.items()} == listed["pools"]
    approve, visited = screen(model, a, b, y[prop], seed, anchor, order[k:], pools, rare)
    grids = {"one": [ceil(k/2)] if k > 1 else [], "three": widths}
    slates, masks = {}, {}
    for budget, grid in grids.items():
        expected = {cols: approve[cols] for w in grid for cols in pools[w][0]}
        assert expected == {tuple(row["cols"]): row["approvals"] for row in listed["screens"][budget]}
        masks[budget] = len(set().union(*(visited[s] for s in expected)))
        for method in ("fraction_gain", "vote_gain", "fraction_lottery", "vote_lottery"):
            sl = []
            for w in grid:
                tie = pools[w][1 if method.endswith("gain") else 2]
                sl.append(next(s for s in tie if approve[s] == max(map(approve.get, tie)))
                          if method.startswith("vote") else tie[0])
            slates[f"{budget}_{method}"] = sl
    assert {key: list(map(list, sl)) for key, sl in slates.items()} == listed["shortlists"]
    supports = sorted({anchor, *(s for sl in slates.values() for s in sl)})
    assert list(map(list, supports)) == sel["supports"] and len(supports) == done["unique_guard_models"]
    inner = np.sort(np.r_[train, prop])
    gp = refits(x, y, inner, guard, supports, seed, folder / "guard.npz", guard=True)
    gs = {s: measures(y[guard], p, rare) for s, p in gp.items()}
    passes = {s: bool(accepted(m, gs[anchor])) for s, m in gs.items()}
    assert [passes[s] for s in supports] == sel["guard_pass"]
    final = {arm: next((s for s in sl if passes[s]), anchor) for arm, sl in slates.items()}
    final["anchor"] = anchor
    assert {arm: list(s) for arm, s in final.items()} == sel["selections"]
    distinct = set(final.values())
    assert len(distinct) == done["unique_outer_models"] and len(outer) == done["n_test"]
    op = refits(x, y, dev, outer, distinct, seed, folder / "predictions.npz")
    os = {s: measures(y[outer], p, rare) for s, p in op.items()}
    fresh = ~np.isin(identity[outer], identity[dev])
    outer_unique = {s: measures(y[outer][fresh], p[fresh], rare) for s, p in op.items()
                    if len(np.unique(y[outer][fresh])) == model.n_classes_}
    guard_fresh = ~np.isin(identity[guard], identity[inner])
    guard_unique = {s: measures(y[guard][guard_fresh], p[guard_fresh], rare) for s, p in gp.items()
                    if len(np.unique(y[guard][guard_fresh])) == model.n_classes_}
    arm_rows = []
    for r in done["rows"]:
        budget, method = r["budget"], r["method"]
        arm = f"{budget}_{method}" if budget != "anchor" else "anchor"
        s, sl = final[arm], slates.get(arm, [])
        failure = not accepted(os[s], os[anchor])
        savings = (1 - len(s)/k) * (not failure)
        assert r["failure"] == failure and r["safe_saved"] == savings
        assert r["k"] == len(s) and r["anchor_k"] == k and r["columns"] == x.shape[1]
        assert r["shortlist_indices"] == [supports.index(c) for c in sl]
        assert r["selected_index"] == supports.index(s) and r["guard_candidates"] == len(sl)
        assert r["masks"] == (masks[budget] if method.startswith("vote") else 0)
        np.testing.assert_allclose(os[s], [r[f] for f in ("auc", "accuracy", "ap") if f in r], atol=1e-12, rtol=0)
        charges = [r[f] for f in ("anchor_pool_s", "vote_s", "guard_s", "final_s")]
        assert min(charges) >= 0 and np.isclose(r["total_s"], sum(charges), rtol=0, atol=1e-12)
        assert (r["vote_s"] > 0) == (method.startswith("vote") and bool(sl))
        reports = dict(dataset=name, fold=fold, budget=budget, method=method, fit_n=len(train),
                       proposal_n=len(prop), guard_n=len(guard), outer_n=len(outer),
                       outer_repeat=int(sum(~fresh)), guard_repeat=int(sum(~guard_fresh)),
                       guard_min_class=int(np.bincount(y[guard]).min()), columns=x.shape[1], anchor_k=k,
                       pool_size=sum(len(pools[w][0]) for w in grids.get(budget, [])),
                       unique_guard_models=len(supports), unique_outer_models=len(distinct),
                       nominated=json.dumps(sl), approvals=json.dumps([approve[c] for c in sl]),
                       guard_pass=json.dumps([passes[c] for c in sl]), guard_candidates=len(sl),
                       mask_calls=r["masks"], chosen=json.dumps(s), k=len(s), fallback=s == anchor,
                       failure=bool(failure), safe_saved=savings, retained=len(s)/x.shape[1],
                       auc=os[s][0], accuracy=os[s][1], anchor_guard_auc=gs[anchor][0],
                       anchor_guard_acc=gs[anchor][1], nominee_guard_auc=json.dumps([gs[c][0] for c in sl]),
                       nominee_guard_acc=json.dumps([gs[c][1] for c in sl]), total_s=r["total_s"],
                       vote_s=r["vote_s"], guard_s=r["guard_s"], final_s=r["final_s"],
                       anchor_pool_s=r["anchor_pool_s"])
        if outer_unique:
            reports.update(newrow_auc=outer_unique[s][0], newrow_acc=outer_unique[s][1],
                           newrow_fail=not accepted(outer_unique[s], outer_unique[anchor]))
        if guard_unique:
            reports["newrow_guard_pass"] = json.dumps([bool(accepted(guard_unique[c], guard_unique[anchor])) for c in sl])
        arm_rows.append(reports)
    assert len(arm_rows) == len(set(final)) == 9
    learned_rows = learned_fold(name, fold, data, order, anchor, approve, rare) if check_learned else []
    return arm_rows, learned_rows


def learned_fold(name, fold, data, order, anchor, approvals, rare):
    x, y, train, prop, guard, dev, outer, _, _ = data
    seed = 20421 + fold
    folder = LEARNED / name / f"fold-{fold}"
    nominations = read(folder / "nominations.json")
    selection = read(folder / "selection.json")
    receipt = read(folder / "completed.json")
    assert nominations["protocol"] == sha(LEARNED / "protocol.json")
    assert nominations["pool"] == sha(STUDY / name / f"fold-{fold}" / "shortlists.json")
    assert selection["nominations"] == sha(folder / "nominations.json")
    assert selection["guard_sha"] == sha(folder / "guard.npz")
    assert receipt["selection"] == sha(folder / "selection.json")
    assert receipt["predictions"] == sha(folder / "predictions.npz")
    tasks = set(read(ROOT / "configs/context_study/prospective.json")["datasets"])
    nominees, pm, maximum = {}, {}, {}
    for source, models in read(MODELS).items():
        for typ in ("size", "size_vote"):
            arm = f"{source}-{typ}"
            donors = [m for m in models if m["model"] == typ and (name == "optdigits" or m["dataset"] == name)]
            assert len(donors) == (14 if name == "optdigits" else 1)
            assert {m["dataset"] for m in donors} == (tasks if name == "optdigits" else {name})
            assert all(set(m["training_tasks"]) == (tasks - {m["dataset"]}) for m in donors)
            assert all(len(m["training_tasks"]) == 13 and name not in m["training_tasks"] for m in donors)
            scores, probabilities = {}, {}
            for s, count in approvals.items():
                deleted = 1 - len(s)/len(anchor)
                features = [deleted, *np.maximum(0, deleted - np.array([.25, .5, .75]))]
                if typ == "size_vote":
                    features.append(1-count/4)
                probs = []
                for m in donors:
                    assert np.all(np.array(m["scales"]) > 0)
                    z = m["intercept"] + sum((v - mean)*coef/scale for v, mean, coef, scale in
                                             zip(features, m["means"], m["coef"], m["scales"], strict=True))
                    probs.append(1 / (1 + np.exp(-z)))
                probabilities[s] = np.mean(probs)
                scores[s] = deleted * (1 - probabilities[s])
            trace = {tuple(r["cols"]): r for r in nominations["traces"][arm]}
            assert set(trace) == set(scores)
            for s, score in scores.items():
                np.testing.assert_allclose([score, probabilities[s]], [trace[s]["value"], trace[s]["failure_probability"]],
                                           rtol=0, atol=1e-13)
            chosen = min(scores, key=lambda s: (-scores[s], sum(order.index(c) for c in s), s)) if scores else anchor
            nominees[arm] = chosen
            pm[arm], maximum[arm] = probabilities.get(chosen, np.nan), scores.get(chosen, np.nan)
    assert {a: list(s) for a, s in nominees.items()} == nominations["nominees"]
    supports = sorted({anchor, *nominees.values()})
    assert list(map(list, supports)) == selection["supports"]
    gp = refits(x, y, np.sort(np.r_[train, prop]), guard, supports, seed, folder / "guard.npz", guard=True)
    gs = {s: measures(y[guard], p, rare) for s, p in gp.items()}
    passed = {s: bool(accepted(v, gs[anchor])) for s, v in gs.items()}
    assert [passed[s] for s in supports] == selection["passed"]
    finals = {a: s if passed[s] else anchor for a, s in nominees.items()}
    assert {a: list(s) for a, s in finals.items()} == selection["final"]
    outer_sets = sorted({anchor, *finals.values()})
    assert list(map(list, outer_sets)) == receipt["deployment_supports"]
    op = refits(x, y, dev, outer, outer_sets, seed, folder / "predictions.npz")
    os = {s: measures(y[outer], p, rare) for s, p in op.items()}
    rows = []
    for r in receipt["rows"]:
        arm = r["source"] + "-" + r["method"]
        q, s = nominees[arm], finals[arm]
        breach = not accepted(os[s], os[anchor])
        saved = (1 - len(s)/len(anchor)) * (not breach)
        assert r["nominee"] == supports.index(q) and r["final"] == outer_sets.index(s)
        assert r["accepted"] == (q != anchor and passed[q])
        assert r["guard_candidates"] == int(q != anchor)
        assert (r["nominee_k"], r["k"], r["anchor_k"]) == (len(q), len(s), len(anchor))
        assert r["failure"] == breach and r["safe_saved"] == saved
        np.testing.assert_allclose(os[s], [r[f] for f in ("auc", "accuracy", "ap") if f in r],
                                   atol=1e-12, rtol=0)
        assert r["total_s"] >= r["vote_seconds"] >= 0
        rows.append(dict(dataset=name, fold=fold, source=r["source"], method=r["method"],
                         candidates_scored=len(approvals), nominee=json.dumps(q), nominee_k=len(q),
                         nominal_failure_p=pm[arm], predicted_value=maximum[arm], chosen=json.dumps(s),
                         k=len(s), anchor_k=len(anchor), accepted=s != anchor, guard_candidates=int(q != anchor),
                         unique_guard_models=len(supports), unique_outer_models=len(outer_sets),
                         failure=bool(breach), safe_saved=saved, auc=os[s][0], accuracy=os[s][1],
                         total_s=r["total_s"], vote_seconds=r["vote_seconds"]))
    return rows


def learned_aggregates(df):
    data = df[df.dataset != "optdigits"]
    for row in pd.read_csv(LEARNED / "paired-effects.csv").itertuples():
        sub = data[data.source == row.source]
        tab = sub.pivot(index=["dataset", "fold"], columns="method", values=row.metric).astype(float)
        d = tab.size_vote - tab["size"]
        task = d.groupby("dataset").mean()
        sim = np.random.default_rng(33119).choice(task.to_numpy(), size=(30000, len(task))).mean(axis=1)
        np.testing.assert_allclose([task.mean(), *np.percentile(sim, [2.5, 97.5])],
                                   [row.delta, row.ci_low, row.ci_high], rtol=0, atol=1e-12)
        for col, count in (("chosen", row.changed_finals), ("nominee", row.changed_nominations),
                           ("nominee_k", row.changed_widths)):
            pairs = sub.pivot(index=["dataset", "fold"], columns="method", values=col)
            assert pairs["size"].ne(pairs.size_vote).sum() == count


def aggregates(df):
    data = df[df.dataset != "optdigits"].rename(columns={"mask_calls": "masks"})
    # Verify totals for each arm, including modal fallback and all no-op folds.
    expected = pd.read_csv(STUDY / "summary.csv").set_index(["budget", "method"]).sort_index()
    fields = [c for c in expected.columns if c not in ("outer_failures", "fallbacks")]
    means = data.groupby(["dataset", "budget", "method"])[fields].mean().groupby(["budget", "method"]).mean()
    np.testing.assert_allclose(means, expected[fields], atol=1e-12, rtol=0)
    counts = data.groupby(["budget", "method"])[["failure", "fallback"]].sum()
    np.testing.assert_array_equal(counts, expected[["outer_failures", "fallbacks"]])
    # Verify the serialized per-fold comparison first. The reporting code read this CSV
    # back through pandas, which can perturb floating-point tie signs after averaging.
    index = ["dataset", "fold", "budget", "method"]
    serialized = pd.read_csv(STUDY / "per-fold.csv").set_index(index).sort_index()
    rebuilt = df.rename(columns={"mask_calls": "masks"}).set_index(index).sort_index()
    assert serialized.index.equals(rebuilt.index)
    for field in ("k", "anchor_k", "columns", "failure", "safe_saved", "guard_candidates",
                  "masks", "anchor_pool_s", "vote_s", "guard_s", "final_s", "total_s", "auc", "accuracy"):
        np.testing.assert_allclose(rebuilt[field].astype(float), serialized[field].astype(float),
                                   atol=1e-12, rtol=0, err_msg=field)
    published = serialized.reset_index()
    published = published[published.dataset != "optdigits"].copy()
    published["fallback"] = published.k == published.anchor_k
    published["retained"] = published.k / published["columns"]
    for row in pd.read_csv(STUDY / "paired-effects.csv").itertuples():
        sub = data[data.budget == row.budget]
        tab = sub.pivot(index=["dataset", "fold"], columns="method", values=row.metric).astype(float)
        delta = tab["vote_" + row.tie] - tab["fraction_" + row.tie]
        task = delta.groupby("dataset").mean()
        rng = np.random.default_rng(33119)
        sim = rng.choice(task.to_numpy(), (30000, len(task)), replace=True).mean(axis=1)
        np.testing.assert_allclose([task.mean(), *np.percentile(sim, [2.5, 97.5])],
                                   [row.delta, row.ci95lo, row.ci95hi], rtol=0, atol=1e-12)
        sub = published[published.budget == row.budget]
        tab = sub.pivot(index=["dataset", "fold"], columns="method", values=row.metric).astype(float)
        d = tab["vote_" + row.tie] - tab["fraction_" + row.tie]
        dt = d.groupby("dataset").mean()
        assert (int((dt > 0).sum()), int((dt < 0).sum()), int((d > 0).sum()), int((d < 0).sum())) == (
            row.pos_tasks, row.neg_tasks, row.pos_folds, row.neg_folds), (row.budget, row.tie, row.metric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Audit one dataset without writing the full output")
    parser.add_argument("--learned", action="store_true", help="Also audit the subsequently defined learned rule")
    opts = parser.parse_args()
    for rel, stamp in read(STUDY / "protocol.json").items():
        assert sha(ROOT / rel) == stamp, rel
    if opts.learned:
        for rel, stamp in read(LEARNED / "protocol.json").items():
            assert sha(ROOT / rel) == stamp, rel
    names = read(ROOT / "configs/context_study/prospective.json")["datasets"]
    eligible = pd.read_csv(ELIGIBLE).set_index(["dataset", "fold"]).eligible
    assert len(eligible) == 42 and eligible.sum() == 41 and len(names) == 14
    rows, learned = [], []
    for name in ["optdigits", *names]:
        if opts.dataset and opts.dataset != name:
            continue
        for fold in (range(1) if name == "optdigits" else range(3)):
            data = partitions(name, fold)
            if name != "optdigits":
                assert bool(eligible[name, fold]) == (data is not None)
            if data is not None:
                results, extra = audit_fold(name, fold, 100 if name == "optdigits" else names.index(name), data,
                                            opts.learned)
                rows.extend(results)
                learned.extend(extra)
        print(f"{name}: {len(rows)} verified arm-folds", flush=True)
    if not opts.dataset:
        df = pd.DataFrame(rows)
        assert len(df) == 42 * 9 and not df.duplicated(["dataset", "fold", "budget", "method"]).any()
        aggregates(df)
        df.to_csv(HERE / "operating-guard-audit.csv", index=False)
        if opts.learned:
            extra = pd.DataFrame(learned)
            assert len(extra) == 42*4 and not extra.duplicated(["dataset", "fold", "source", "method"]).any()
            learned_aggregates(extra)
            extra.to_csv(HERE / "operating-guard-audit-learned.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
