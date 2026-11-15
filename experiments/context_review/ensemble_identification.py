"""Exact finite-sample refit metric sets for fitted trees and a common mask transcript."""
from collections import Counter
from fractions import Fraction
from itertools import combinations_with_replacement, product
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


EVIDENCE = Path(__file__).resolve().parents[2] / "thesis-paper/evidence/ensemble_cells"


def study_data():
    domain = np.array(list(product(range(2), range(2), range(3))))
    x, y = [], []
    for d, g, r in domain:
        n = 40 if d == 0 else 20
        positives = (3, 15)[g] if d == 0 else (8, 14, 19)[r]
        x.extend([(d, g, r)] * n)
        y.extend([1] * positives + [0] * (n - positives))
    evaluation = [(0, 1, 2), (0, 1, 1), (1, 1, 2), (0, 1, 0), (0, 0, 1), (1, 0, 0)]
    rows = [list(map(tuple, domain)).index(row) for row in evaluation]
    return domain, np.array(x), np.array(y), rows, np.array([1, 1, 1, 0, 0, 0])


def score_transcript(model, domain, background):
    masks = list(product((False, True), repeat=domain.shape[1]))
    scores = [model.predict_proba(np.where(mask, domain, background))[:, 1] for mask in masks]
    return list(zip(*scores)), masks


def conditional_laws(groups, known_histogram):
    """Enumerate integer counts, grouping rows indistinguishable for score assignment."""
    local_tables = []
    for (label, allowed), n in groups.items():
        local_tables.append([(label, Counter(choice)) for choice in combinations_with_replacement(allowed, n)])
    for assignment in product(*local_tables):
        counts = (Counter(), Counter())
        for label, bins in assignment:
            counts[label].update(bins)
        if known_histogram is None or counts[0] + counts[1] == known_histogram:
            yield counts


def counted_metrics(counts, decisions):
    negatives, positives = counts
    twice_wins = sum(npos * nneg * (2 * int(s > t) + int(s == t))
                     for s, npos in positives.items() for t, nneg in negatives.items())
    correct = sum(count for label, bins in enumerate(counts)
                  for score, count in bins.items() if decisions[score] == label)
    p, n = positives.total(), negatives.total()
    return Fraction(twice_wins, 2 * p * n), Fraction(correct, p + n)


def counted_average_precision(negatives, positives):
    seen = correct = 0
    total = Fraction(0)
    for score in sorted(set(negatives) | set(positives), reverse=True):
        correct += positives[score]
        seen += positives[score] + negatives[score]
        total += Fraction(positives[score] * correct, positives.total() * seen)
    return total


def check_vector_scores():
    # Equal components in distinct probability vectors exercise the required
    # class-specific pooling of tied ranks.
    scores = np.array([[.5, .25, .25], [.5, .375, .125], [.25, .25, .5], [.125, .625, .25]])
    labels = np.array([0, 0, 1, 1, 2, 2])
    allowed = [(0, 1), (1, 2), (2, 3), (1, 3), (0, 2), (2, 3)]
    for selected in product(*allowed):
        matrix = scores[list(selected)]
        aucs = []
        for c in range(matrix.shape[1]):
            pos = labels == c
            bins = (Counter(matrix[~pos, c]), Counter(matrix[pos, c]))
            # The first result depends only on ranks and ties, so the class
            # decisions in this auxiliary binary calculation can be arbitrary.
            aucs.append(counted_metrics(bins, {s: 0 for s in matrix[:, c]})[0])
            assert np.isclose(float(counted_average_precision(*bins)), average_precision_score(pos, matrix[:, c]))
        macro = sum(aucs) / len(aucs)
        assert np.isclose(float(macro), roc_auc_score(labels, matrix, multi_class="ovr", average="macro"))
    return 2 ** len(allowed)


def metrics_from_predictions(y, scores, decisions):
    npos = int(y.sum())
    auc_scale = 2 * npos * (len(y) - npos)
    auc = round(roc_auc_score(y, scores) * auc_scale)
    accuracy = round(accuracy_score(y, [decisions[s] for s in scores]) * len(y))
    return Fraction(auc, auc_scale), Fraction(accuracy, len(y))


def wilcoxon_bounds_check(counts, histogram, auc):
    # With fixed pooled frequencies, the pooled midranks are fixed for every world.
    below = 0
    ranks = {}
    for score, n in sorted(histogram.items()):
        ranks[score] = below + Fraction(n + 1, 2)
        below += n
    npos, nneg = counts[1].total(), counts[0].total()
    rank_sum = sum(ranks[s] * k for s, k in counts[1].items())
    assert (rank_sum - Fraction(npos * (npos + 1), 2)) / (npos * nneg) == auc


def optimize_histogram(groups, histogram, decisions, auc_weight, accuracy_weight):
    """Maximize a signed linear objective over fixed pooled margins by integral transport."""
    supply = Counter()
    for (label, _), count in groups.items():
        supply[label] += count
    npos, nneg = supply[1], supply[0]
    size = npos + nneg
    graph = nx.DiGraph()
    below = 0
    twice_rank = {}
    for score, count in sorted(histogram.items()):
        twice_rank[score] = 2 * below + count + 1
        graph.add_node(("score", score), demand=count)
        below += count
    for group, ((label, allowed), count) in enumerate(groups.items()):
        graph.add_node(group, demand=-count)
        for score in allowed:
            if score not in histogram:
                continue
            rank_cost = auc_weight * label * twice_rank[score] * size
            hit_cost = accuracy_weight * int(decisions[score] == label) * 2 * npos * nneg
            graph.add_edge(group, ("score", score), capacity=count, weight=-int(rank_cost + hit_cost))
    _, flow = nx.network_simplex(graph)
    counts = (Counter(), Counter())
    for group, (label, _) in enumerate(groups):
        for (_, score), n in flow[group].items():
            counts[label][score] += n
    return counted_metrics(counts, decisions)


def audit(name, estimator):
    domain, x, y, row_ids, outer_y = study_data()
    background = np.median(x, axis=0)
    frozen = clone(estimator).fit(x, y)
    reduced = clone(estimator).fit(x[:, 1:], y)
    transcript, masks = score_transcript(frozen, domain, background)
    scores = list(map(float, reduced.predict_proba(domain[:, 1:])[:, 1]))
    predicted = reduced.predict(domain[:, 1:])
    decisions = {s: int(cls) for s, cls in zip(scores, predicted)}
    assert all(decisions[s] == cls for s, cls in zip(scores, predicted))
    raw_choices = [[j for j, t in enumerate(transcript) if t == transcript[i]] for i in row_ids]
    allowed = [tuple(sorted({scores[j] for j in indices})) for indices in raw_choices]
    groups = Counter(zip(outer_y.tolist(), allowed))
    histogram = Counter(scores[i] for i in row_ids)
    full_scores = [float(transcript[i][-1]) for i in row_ids]
    full_classes = frozen.predict(domain[row_ids])
    anchor = metrics_from_predictions(outer_y, full_scores, dict(zip(full_scores, full_classes)))
    rows, summary = [], []
    for side, fixed_hist in (("rows_only", None), ("pooled_histogram", histogram)):
        exact = set()
        for counts in conditional_laws(groups, fixed_hist):
            measured = counted_metrics(counts, decisions)
            exact.add(measured)
            if fixed_hist is not None:
                wilcoxon_bounds_check(counts, fixed_hist, measured[0])

        # Independent check visits individual *raw feature rows* and scores each
        # resulting full evaluation set using sklearn's public metrics.
        reference = set()
        raw_worlds = 0
        for selected_rows in product(*raw_choices):
            actual = [scores[j] for j in selected_rows]
            if fixed_hist is not None and Counter(actual) != fixed_hist:
                continue
            raw_worlds += 1
            reference.add(metrics_from_predictions(outer_y, actual, decisions))
        assert exact == reference and len(exact) > 1
        if fixed_hist is None:
            for opt, aggregator in ((max, max), (min, min)):
                choose_neg = min if opt is max else max
                extreme = [opt(v) if pos else choose_neg(v) for pos, v in zip(outer_y, allowed)]
                corner = metrics_from_predictions(outer_y, extreme, decisions)
                assert corner == tuple(aggregator(v) for v in zip(*exact))
        else:
            for u, v in product(range(-2, 3), repeat=2):
                if u or v:
                    auc, accuracy = optimize_histogram(groups, histogram, decisions, u, v)
                    assert (auc, accuracy) in exact
                    assert u * auc + v * accuracy == max(u * a + v * c for a, c in exact)
        for auc, acc in sorted(exact):
            rows.append(dict(model=name, constraint=side, auc_fraction=str(auc), accuracy_fraction=str(acc)))
        passage = {all(v + 1e-12 >= .97 * a for v, a in zip(pair, anchor)) for pair in exact}
        summary.append(dict(model=name, constraint=side, feasible_worlds=raw_worlds, distinct_metric_pairs=len(exact),
                            min_auc=float(min(z[0] for z in exact)), max_auc=float(max(z[0] for z in exact)),
                            min_accuracy=float(min(z[1] for z in exact)), max_accuracy=float(max(z[1] for z in exact)),
                            both_pass_and_fail=len(passage) == 2))
    details = dict(training_rows=len(x), domain=domain.tolist(), masks=masks, median=background.tolist(),
                   frozen_scores=np.asarray(transcript).tolist(), outer_row_ids=row_ids, outer_labels=outer_y.tolist(),
                   reduced_scores=scores, feasible_scores_by_row=allowed, anchor_metrics=list(map(str, anchor)),
                   distinct_transcripts=len(set(transcript)), distinct_reduced_scores=len(set(scores)))
    unique_transcripts = sorted(set(transcript))
    details["compatible_score_counts"] = [len({s for t, s in zip(transcript, scores) if t == observation})
                                          for observation in unique_transcripts]
    details["verified_integer_cost_directions"] = 24
    if isinstance(frozen, DecisionTreeClassifier):
        details["tree_depth"] = frozen.tree_.max_depth
    else:
        trees = frozen.get_booster().get_dump()
        details["tree_count"] = len(trees)
        details["max_fitted_depth"] = max(len(line) - len(line.lstrip("\t")) for tree in trees for line in tree.splitlines())
    return rows, summary, details


def main():
    learners = {
        "cart": DecisionTreeClassifier(max_depth=3, random_state=90),
        "xgboost": XGBClassifier(n_estimators=9, max_depth=3, learning_rate=.4, min_child_weight=0,
                                 reg_lambda=0, reg_alpha=0, tree_method="hist", n_jobs=1, random_state=90),
    }
    pairs, summaries, info = [], [], {}
    for name, learner in learners.items():
        rows, summary, info[name] = audit(name, learner)
        pairs.extend(rows)
        summaries.extend(summary)
    info["multiclass_vector_assignments_checked"] = check_vector_scores()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_csv(EVIDENCE / "metric-sets.csv", index=False)
    frame = pd.DataFrame(summaries)
    frame.to_csv(EVIDENCE / "summary.csv", index=False)
    (EVIDENCE / "examples.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

