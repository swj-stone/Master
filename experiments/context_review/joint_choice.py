"""Joint tolerance decisions when two candidates share the same outer-row assignment."""
from collections import defaultdict
from contextlib import ExitStack
from functools import reduce
from hashlib import sha256
from io import BytesIO
from itertools import combinations, product
from math import lcm
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

from experiments.context_review import transcript_information as ti
from experiments.context_review import transcript_permutations as tp


DEST = tp.ROOT / "thesis-paper/evidence/joint_choice"
DESIGN = tp.ROOT / "configs/context_review/joint_choice.json"
STATES = {(False, False), (False, True), (True, False), (True, True)}


def label_assignments(group, y):
    """Enumerate distinct labelings of the fixed feature rows, modulo equal labels."""
    labels, counts = np.unique(y[group], return_counts=True)
    assigned = np.empty(len(group), dtype=int)

    def fill(k, free):
        if k == len(labels) - 1:
            assigned[list(free)] = labels[k]
            yield assigned.copy()
            return
        for taken in combinations(free, counts[k]):
            assigned[list(taken)] = labels[k]
            yield from fill(k + 1, tuple(i for i in free if i not in taken))

    yield from fill(0, tuple(range(len(group))))


def assigned_permutation(y, codes, assignments):
    """Construct a common donor permutation that realizes the stipulated labels."""
    assigned = y.copy()
    for group, labels in assignments:
        assigned[group] = labels
    perm = np.arange(len(y))
    for code in np.unique(codes):
        ids = np.flatnonzero(codes == code)
        for label in np.unique(y[ids]):
            perm[ids[y[ids] == label]] = ids[assigned[ids] == label]
    assert np.array_equal(codes[perm], codes) and np.array_equal(np.sort(perm), np.arange(len(y)))
    return perm


def score_basis(y, probs, denominator):
    """Integer numerators of tied-rank AUC terms and numbers of correct predictions."""
    n, k = probs.shape
    targets = [1] if k == 2 else list(range(k))
    # Python integers retain the exact common denominator even for multiclass AUC.
    ranks = np.zeros((n, k), dtype=object)
    constant = 0
    for c in targets:
        npos = int(np.count_nonzero(y == c))
        weight = denominator // (npos * (n - npos))
        doubled = (2 * rankdata(probs[:, c])).astype(int).tolist()
        ranks[:, c] = [weight * r for r in doubled]
        constant += weight * npos * (npos + 1)
    correct = (probs.argmax(axis=1)[:, None] == np.arange(k)).astype(int)
    return ranks, correct, constant


class Decision:
    def __init__(self, y, predictions, ref, common):
        ranks, correct, offset = score_basis(y, predictions, common)
        n, k = predictions.shape
        rank_gap, acc_gap = 100 * ranks - 97 * ref[0], 100 * correct - 97 * ref[1]
        rows = np.arange(n)
        self.values = (rank_gap, acc_gap)
        self.base = (int(rank_gap[rows, y].sum()) - 3 * offset, int(acc_gap[rows, y].sum()))
        fractions = (200 * (1 if k == 2 else k) * common, 100 * n)
        # ceil(-den/10^12) is the integral numerator threshold including the archive's slack.
        self.floor = tuple(-(den // 10**12) for den in fractions)
        self.factual = all(a >= b for a, b in zip(self.base, self.floor, strict=True))
        self.scaled = tuple(np.asarray(x / d, dtype=np.float64) for x, d in zip(self.values, fractions, strict=True))
        self.actual = np.asarray(self.base, dtype=np.float64) / np.asarray(fractions, dtype=np.float64)

    def state(self, permutation, y):
        n = len(y)
        return all(base + int((obj[permutation, y] - obj[np.arange(n), y]).sum()) >= floor
                   for obj, base, floor in zip(self.values, self.base, self.floor, strict=True))


def allocations_for_group(group, y, values):
    original = [v[group, y[group]].sum() for v in values]
    # Distinct equivalent assignments can yield the same four decision numerators.
    possibilities = {}
    for labels in label_assignments(group, y):
        delta = tuple(int(v[group, labels].sum() - initial) for v, initial in zip(values, original, strict=True))
        if delta not in possibilities:
            possibilities[delta] = labels
    return possibilities


def exact_point(y, observed, models, fixed):
    """Exhaust all decision-relevant group contributions, pruning only known joint states."""
    zero = (0, 0, 0, 0)
    identity = np.arange(len(y))
    original = tuple(m.factual for m in models)
    found = {original: identity}
    if all(fixed):
        return found
    codes = tp.partition(observed)
    values = models[0].values + models[1].values
    items = []
    for group in np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1]):
        if len(group) == 1 or np.all(y[group] == y[group[0]]):
            continue
        if all(np.all(v[group] == v[group[0]]) for v in values):
            continue
        options = allocations_for_group(group, y, values)
        if len(options) > 1:
            items.append((group, list(options.items())))
    items.sort(key=lambda it: len(it[1]), reverse=True)
    n = len(items)
    lower, upper = [zero] * (n + 1), [zero] * (n + 1)
    for i in reversed(range(n)):
        axes = list(zip(*(item[0] for item in items[i][1]), strict=True))
        lower[i] = tuple(lo + min(ax) for lo, ax in zip(lower[i + 1], axes, strict=True))
        upper[i] = tuple(hi + max(ax) for hi, ax in zip(upper[i + 1], axes, strict=True))
    floors = models[0].floor + models[1].floor
    base = tuple(a - b for a, b in zip(models[0].base + models[1].base, floors, strict=True))
    chosen = []

    def visit(i, gaps):
        possibilities = []
        for model in range(2):
            j = 2 * model
            if fixed[model]:
                possibilities.append([original[model]])
            elif all(gaps[k] + lower[i][k] >= 0 for k in (j, j + 1)):
                possibilities.append([True])
            elif any(gaps[k] + upper[i][k] < 0 for k in (j, j + 1)):
                possibilities.append([False])
            else:
                possibilities.append([False, True])
        feasible_envelope = set(product(*possibilities))
        if feasible_envelope <= found.keys():
            return
        if len(feasible_envelope) == 1:
            state = next(iter(feasible_envelope))
            perm = assigned_permutation(y, codes, chosen)
            assert tuple(m.state(perm, y) for m in models) == state
            found[state] = perm
            return
        group, choices = items[i]
        for step, labels in choices:
            chosen.append((group, labels))
            visit(i + 1, tuple(a + b for a, b in zip(gaps, step, strict=True)))
            chosen.pop()
            if feasible_envelope <= found.keys():
                break

    visit(0, base)
    return found


def code(state):
    return "".join(str(int(s)) for s in state)


def fractional_search(models, y, wanted):
    """Seek feasible assignments near strict interiors; only validated permutations are evidence."""
    n, classes = models[0].scaled[0].shape
    identity = np.arange(n)
    # x_ij assigns label j to fixed feature vector i. The LP keeps each class size fixed.
    variables = np.arange(n * classes)
    row = np.repeat(identity, classes)
    label = np.tile(np.arange(classes), n)
    incidence = csr_matrix((np.ones(2 * len(row)), (np.r_[row, n+label], np.tile(variables, 2))),
                           shape=(n + classes, len(variables)))
    totals = np.r_[np.ones(n), np.bincount(y)]
    # A pass imposes both inequalities. For a failure either violated metric is sufficient.
    alternatives = [((0, 1),) if pass_bit else ((0,), (1,)) for pass_bit in wanted]
    for chosen in product(*alternatives):
        inequalities, bounds = [], []
        for model, pass_bit, metrics in zip(models, wanted, chosen, strict=True):
            sign = 1 if pass_bit else -1
            for j in metrics:
                values = model.scaled[j]
                constant = model.actual[j] - values[identity, y].sum()
                inequalities.append(-sign * n * values.ravel())
                bounds.append(sign * n * constant)
        # Maximize the smallest signed slack. A relaxed solution may be fractional.
        # Only actual row permutations survive the independent integer-tolerance check below.
        matrix = csr_matrix(np.column_stack([np.asarray(inequalities), np.ones(len(bounds))]))
        equal = csr_matrix((incidence.data, incidence.indices, incidence.indptr),
                           shape=(n + classes, len(variables)+1))
        solution = linprog(np.r_[np.zeros(n * classes), -1.], A_ub=matrix, b_ub=bounds,
                           A_eq=equal, b_eq=totals, bounds=[(0, 1)]*(n*classes) + [(None, None)],
                           method="highs", options={"time_limit": 25})
        if not solution.success:
            continue
        x = solution.x[:-1].reshape(n, classes)
        fixed = x.max(axis=1) > 1 - 1e-7
        labels = x.argmax(axis=1)
        missing = np.bincount(y, minlength=classes) - np.bincount(labels[fixed], minlength=classes)
        if np.any(missing < 0):
            continue
        donors = np.repeat(np.arange(classes), missing)
        uncertain = np.flatnonzero(~fixed)
        if len(donors):
            _, match = linear_sum_assignment(x[uncertain][:, donors], maximize=True)
            labels[uncertain] = donors[match]
        p = assigned_permutation(y, np.zeros(n, dtype=int), [(identity, labels)])
        if tuple(m.state(p, y) for m in models) == wanted:
            return p
    return None


def compare_fold(archive, key, rec, point_invariant, witnesses, sealed=None):
    data = np.load(BytesIO(archive.read(key + ".npz")))
    y, anchor = data["y"], data["anchor"]
    n, k = anchor.shape
    counts = np.bincount(y).tolist()
    common = reduce(lcm, (s * (n - s) for s in (counts[1:] if k == 2 else counts)))
    proposals = rec["proposals"]
    raw = [data[f"s{i}"] for i in range(len(proposals))]
    ref = score_basis(y, anchor, common)
    decisions = [Decision(y, p, ref, common) for p in raw]
    for p, model in zip(raw, decisions, strict=True):
        direct, _ = ti.paired_gap(y, p, anchor)
        np.testing.assert_allclose(model.actual, direct - 1e-12, rtol=0, atol=1e-13)
        assert model.factual == bool(np.all(direct >= 0))
    donor = tp.ROOT / "results/context_review/refit-target" / key / "selection.json"
    assert tp.digest(donor) == rec["selection"]
    anchor_width = len(tp.rules.read(donor)["anchor"])
    row_ids = np.arange(n)
    results, examples, world_ids, distinct = [], {}, {}, []
    world_pool = [row_ids]
    for w in witnesses:
        perm = row_ids.copy()
        perm[w["positions"]] = w["donors"]
        world_pool.append(perm)
    # These are feasible development-only worlds shared by every pair in the fold.
    pool_states = (None if sealed is not None
                   else np.array([[m.state(p, y) for m in decisions] for p in world_pool]))
    blocks = np.array_split(row_ids, max(1, int(np.ceil(n / 48))))
    masks = data["codes"].T
    for a, b in combinations(range(len(proposals)), 2):
        width = len(proposals[a]["columns"])
        if width != len(proposals[b]["columns"]):
            continue
        d = 1 - width / anchor_width
        candidates = (decisions[a], decisions[b])
        observed = masks[:, sorted(set(proposals[a]["point"] + proposals[b]["point"]))]
        point = exact_point(y, observed, candidates, (a in point_invariant, b in point_invariant))
        dev = dict(point)
        if sealed is None:
            for perm, flags in zip(world_pool, pool_states[:, [a, b]], strict=True):
                dev.setdefault(tuple(flags), perm)
        else:
            perms, references = sealed
            for state in sorted(STATES):
                ix = references[a, b, "development"][code(state)]
                assert ix >= 0
                perm = perms[ix]
                assert tuple(m.state(perm, y) for m in candidates) == state
                dev[state] = perm
        if len(dev) < 4:
            # Each optimized direction only supplies witnesses. It never rules a state out.
            for flags in sorted(STATES - dev.keys()):
                for left in ((candidates[0].scaled[0], candidates[0].scaled[1]) if not flags[0]
                             else (sum(candidates[0].scaled),)):
                    for right in ((candidates[1].scaled[0], candidates[1].scaled[1]) if not flags[1]
                                  else (sum(candidates[1].scaled),)):
                        for ratio in (1, .25, 4):
                            obj = left * (1 if flags[0] else -1) + right * (ratio if flags[1] else -ratio)
                            perm = tp.extremizer(y, obj, blocks, True)
                            state = tuple(m.state(perm, y) for m in candidates)
                            dev.setdefault(state, perm)
                            if flags in dev:
                                break
                        if flags in dev:
                            break
                    if flags in dev:
                        break
        for flags in sorted(STATES - dev.keys()):
            perm = fractional_search(candidates, y, flags)
            if perm is not None:
                dev[flags] = perm
        for observer, states in (("development", dev), ("both_points", point)):
            if sealed is not None:
                refs = references[a, b, observer]
                assert set(map(code, states)) == {bits for bits, ix in refs.items() if ix >= 0}
                states = {state: perms[refs[code(state)]] for state in states}
            diffs = sorted({int(u) - int(v) for u, v in states})
            exact = observer == "both_points" or len(states) == 4
            if exact:
                lower, upper = min(diffs), max(diffs)
            else:
                lower, upper = -1, 1
            flags_a, flags_b = ({s[j] for s in states} for j in range(2))
            if len(flags_a) == len(flags_b) == 2 and len(diffs) == 1 and exact:
                kind = "joint_tie"
            elif diffs == [-1, 0, 1]:
                kind = "both_orders"
            elif exact and len(diffs) > 1:
                kind = "one_order"
            else:
                kind = "constant" if exact else "open"
            record = dict(dataset=rec["dataset"], fold=rec["fold"], a=a, b=b, kept=width,
                          anchor=anchor_width, observer=observer, joint=" ".join(sorted(map(code, states))),
                          joint_exact=exact, gap_units=" ".join(map(str, diffs)),
                          gap_exact=exact or diffs == [-1, 0, 1],
                          min_witness=d * min(diffs), max_witness=d * max(diffs),
                          lower_bound=d * lower, upper_bound=d * upper, kind=kind)
            for state in sorted(STATES):
                world_id = -1
                if sealed is not None:
                    world_id = refs[code(state)]
                elif state in states:
                    perm16 = states[state].astype("<u2")
                    assert np.array_equal(states[state], perm16)
                    key_bytes = perm16.tobytes()
                    if key_bytes not in world_ids:
                        world_ids[key_bytes] = len(distinct)
                        distinct.append(perm16)
                    world_id = world_ids[key_bytes]
                record["world_" + code(state)] = world_id
            results.append(record)
            if (observer, kind) in examples:
                continue
            evidence = {}
            for state, p in states.items():
                assert np.array_equal(np.sort(p), row_ids)
                if observer == "both_points":
                    np.testing.assert_array_equal(observed, observed[p])
                records = [ti.paired_gap(y, raw[i][p], anchor[p]) for i in (a, b)]
                assert tuple(bool(np.all(gaps >= 0)) for gaps, _ in records) == state
                moved = np.flatnonzero(p != row_ids)
                evidence[code(state)] = dict(positions=moved.tolist(), donors=p[moved].tolist(),
                                             a=records[0][1]["candidate"], b=records[1][1]["candidate"],
                                             anchor=records[0][1]["anchor"])
            examples[observer, kind] = dict(pair=record, witnesses=evidence)
    return results, examples, np.stack(distinct) if distinct else None


def replay(record_worlds=False):
    path = tp.DEST / "row-partitions.zip"
    manifest = tp.rules.read(tp.DEST / "manifest.json")
    assert tp.digest(path) == manifest["archive"]
    marginal = pd.read_csv(ti.DEST / "proposals.csv").query("observer == 'outer_point'")
    assert len(marginal) == 704 and set(marginal.disposition) == {"witness", "invariant"}
    archived = defaultdict(list)
    for w in tp.rules.read(ti.DEST / "witnesses.json"):
        if w["observer"] == "proposal_only":
            archived[w["dataset"], w["fold"]].append(w["alternative"])
    tables, exemplars = [], {}
    DEST.mkdir(parents=True, exist_ok=True)
    saved_worlds = {}
    witness_path = DEST / "witnesses.zip"
    signature = dict(row_partitions=manifest["archive"],
                     design=sha256(json.dumps(tp.rules.read(DESIGN), sort_keys=True).encode("utf8")).hexdigest())
    if not record_worlds:
        assert tp.rules.read(DEST/"receipt.json") == {**signature, "witnesses":tp.digest(witness_path)}
    with ExitStack() as stack:
        archive = stack.enter_context(ZipFile(path))
        frozen = None if record_worlds else stack.enter_context(ZipFile(witness_path))
        manifest_ids = {} if frozen is None else json.loads(frozen.read("world-index.json"))
        for key, rec in sorted(manifest["folds"].items()):
            fixed = set(marginal[(marginal.dataset == rec["dataset"]) & (marginal.fold == rec["fold"])
                                 & (marginal.disposition == "invariant")].proposal)
            sealed = None
            if frozen is not None:
                lookup = {(r[0], r[1], r[2]): r[3] for r in manifest_ids[key]}
                sealed = np.load(BytesIO(frozen.read(key + ".npy"))), lookup
            rows, examples, permutations = compare_fold(archive, key, rec, fixed,
                                                         archived[rec["dataset"], rec["fold"]], sealed)
            if record_worlds:
                with BytesIO() as out:
                    np.save(out, permutations, allow_pickle=False)
                    saved_worlds[key + ".npy"] = out.getvalue()
                manifest_ids[key] = [[r["a"], r["b"], r["observer"],
                                      {code(s): r["world_" + code(s)] for s in sorted(STATES)}] for r in rows]
            tables.extend(rows)
            for tag, value in examples.items():
                exemplars.setdefault(tag, value)
            print(f"{key}: {len(rows)//2} pairs", flush=True)
    if record_worlds:
        saved_worlds["world-index.json"] = json.dumps(manifest_ids, separators=(",", ":")).encode("utf8")
        pending = witness_path.with_suffix(".tmp")
        with ZipFile(pending, "w") as out:
            for entry, data in saved_worlds.items():
                out.writestr(ZipInfo(entry), data, compress_type=ZIP_DEFLATED)
        pending.replace(witness_path)
        (DEST/"receipt.json").write_text(json.dumps({**signature, "witnesses":tp.digest(witness_path)},
                                                    indent=2)+"\n", encoding="utf8")
    pairs = pd.DataFrame(tables)
    pairs.to_csv(DEST / "pairs.csv", index=False)
    summary = pairs.groupby(["observer", "joint", "joint_exact", "gap_units", "gap_exact", "kind"]).agg(
        pairs=("a", "size"), tasks=("dataset", "nunique")).reset_index()
    summary.to_csv(DEST / "summary.csv", index=False)
    (DEST / "examples.json").write_text(json.dumps(list(exemplars.values()), indent=2) + "\n", encoding="utf8")
    from experiments.context_review.joint_choice_check import case_study
    case_study()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-worlds", action="store_true",
                        help="Rebuild the archive of explicit common row permutations for the reported states")
    replay(parser.parse_args().record_worlds)
