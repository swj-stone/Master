"""Tests for the native statistical selector and the matched permutation game."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from minppi_mofs.baselines._permutation import permutation_scores


def silent(*args, **kwargs):
    pass


def test_min_versus_mean_share_paths_and_match_direct_enumeration():
    p = 8
    names = list(map(str,range(p)))
    calls = Counter()
    def utility(ids):
        calls[ids]+=1
        return sum(i/8 for i in ids) + (4 if 1 in ids and 3 in ids else 0)
    scores,selected,diag = permutation_scores("shapley","native",names,utility,silent)
    assert selected is None and max(calls.values()) == 1
    expected = np.zeros((4,p))
    rng = np.random.default_rng(42)
    for k in range(4):
        included = set()
        for j in rng.permutation(p):
            previous = utility(tuple(sorted(included)))
            included.add(int(j))
            expected[k,j] = utility(tuple(sorted(included)))-previous
    np.testing.assert_array_equal(np.array(diag["sampled_marginals"]),expected)
    np.testing.assert_array_equal(scores, expected.mean(axis=0))
    other,_,compare = permutation_scores("minshap","frozen",names,utility,silent,
                                         evaluate_batch=lambda groups:[utility(x) for x in groups])
    assert diag["permutation_sha256"] == compare["permutation_sha256"]
    np.testing.assert_array_equal(other, expected.min(axis=0))
    assert (other < scores).any()


def test_dummy_projection_leaves_exact_scores_for_full_shared_permutations():
    cols = list("abcdefg")
    def value(t):
        return int(0 in t and 4 in t)*3 + int(1 in t)
    for name in ("minshap","shapley"):
        full,_,diag = permutation_scores(name,"frozen",cols,value,silent)
        active,_,partial = permutation_scores(name,"frozen",cols,value,silent,
                                              active=np.array([0,1,4]),evaluate_batch=lambda ts:[value(t) for t in ts])
        np.testing.assert_array_equal(active,full)
        np.testing.assert_array_equal(partial["sampled_marginals"],diag["sampled_marginals"])
        assert partial["permutation_sha256"] == diag["permutation_sha256"]
        assert partial["oracle_prefixes"] < diag["oracle_prefixes"]
    with pytest.raises(ValueError,match="refit game"):
        permutation_scores("minshap","native",cols,value,silent,active=[0])


def test_native_threshold_uses_paired_loss_variance_and_associated_path():
    names = list("abc")
    base = np.ones(6)*17
    def loss(mask):
        diff = np.ones(6)*(.6*(0 in mask) - .3*(1 in mask))
        if 0 in mask and 2 in mask:
            diff += np.array([2,-2,1,-1,2,-2]) - .2
        return base-diff
    scores,select,info = permutation_scores("minshap","native",names,loss,silent)
    details = np.array(info["sampled_marginals"])
    variances = np.zeros_like(details)
    rng = np.random.default_rng(42)
    for i in range(4):
        mask = set()
        for j in rng.permutation(3):
            prev = loss(mask)
            mask.add(j)
            paired = prev-loss(mask)
            assert paired.mean() == pytest.approx(details[i,j])
            variances[i,j] = np.var(paired,ddof=1)/len(paired)
    np.testing.assert_allclose(info["variance_of_mean"], variances)
    minimum = details.argmin(axis=0)
    expected = np.sqrt(-2*np.log(.05)*variances[minimum,np.arange(3)])
    np.testing.assert_allclose(info["thresholds"], expected)
    np.testing.assert_allclose(scores, details.min(axis=0))
    assert select == [name for name,v,t in zip(names,scores,expected) if v>=t]


def test_native_no_hits_is_an_empty_support_and_interruption_propagates():
    cols = list("ab")
    scores,support,diag = permutation_scores("minshap","native",cols,
                           lambda key:np.ones(5)*(1+len(key)),silent)
    assert support == [] and np.all(scores==-1) and diag["loss_rows"]==5
    def interrupted(mask):
        if len(mask)>1:
            raise TimeoutError
        return 0.
    with pytest.raises(TimeoutError):
        permutation_scores("shapley","native",cols,interrupted,silent)


@pytest.mark.parametrize("mode", ["native", "frozen"])
def test_joint_visit_matches_both_standalone_estimators_without_repeated_oracle_calls(mode):
    from minppi_mofs.baselines._permutation import select_permutation, select_permutation_pair

    cols = list("abcdefgh")
    active = (1, 2, 6) if mode == "frozen" else None
    def value(ids):
        return .3*(1 in ids) + .4*(2 in ids and 6 in ids)
    def losses(ids):
        return np.ones(7) + np.arange(7)*(.015*(3 in ids)-.04*(1 in ids))-.35*(2 in ids)
    ctx = SimpleNamespace(columns=cols, active=active, seed=73, mode=mode, value=value, losses=losses,
                          values=lambda batch: [value(mask) for mask in batch], progress=silent)
    separate = {method: select_permutation(ctx,method) for method in ("minshap","shapley")}
    calls = Counter()
    def native(ids):
        calls[ids] += 1
        return losses(ids), value(ids)
    def scalar(ids):
        calls[ids] += 1
        return value(ids)
    def unexpected(*args):
        pytest.fail("Wrong oracle for a paired game")
    ctx.losses = unexpected
    ctx.value = scalar if mode == "frozen" else unexpected
    ctx.values = (lambda batch: [scalar(ids) for ids in batch]) if mode == "frozen" else unexpected
    both = select_permutation_pair(ctx,joint_values=native if mode=="native" else unexpected)
    assert max(calls.values()) == 1 and all(mask == tuple(sorted(mask)) for mask in calls)
    assert len(calls) == both["minshap"].diagnostics["oracle_prefixes"]
    for method, answer in both.items():
        np.testing.assert_array_equal(answer.scores,separate[method].scores)
        assert answer.selected == separate[method].selected
        assert answer.diagnostics == separate[method].diagnostics


def test_joint_empty_frozen_projection_needs_one_prediction():
    from minppi_mofs.baselines._permutation import select_permutation_pair

    seen = []
    def constant(mask):
        seen.append(mask)
        return 1.
    ctx = SimpleNamespace(columns=["a","b"], active=(), seed=42, mode="frozen",
                          value=constant, values=None, progress=silent)
    both = select_permutation_pair(ctx)
    assert seen == [()]
    for selection in both.values():
        np.testing.assert_array_equal(selection.scores,[0.,0.])


@pytest.mark.parametrize("method", ["minshap", "shapley", "pair"])
def test_native_batching_preserves_order_and_calls_every_distinct_prefix_once(method):
    from minppi_mofs.baselines._permutation import select_permutation, select_permutation_pair

    def loss(mask):
        return np.arange(5.)*(1. if 3 in mask else .5) + len(mask)
    def value(mask):
        return .4*(3 in mask and 5 in mask) + .01*len(mask)
    def observation(mask):
        return (loss(mask),value(mask)) if method=="pair" else loss(mask) if method=="minshap" else value(mask)
    def run(ctx):
        return (select_permutation_pair(ctx,joint_values=observation) if method=="pair"
                else {method:select_permutation(ctx,method)})
    ctx=SimpleNamespace(columns=list("abcdefghijk"),mode="native",seed=42,
                        losses=loss,value=value,values=None,progress=silent)
    old=run(ctx)
    all_masks, sizes=[],[]
    def ordered_batch(masks):
        all_masks.extend(masks)
        sizes.append(len(masks))
        # Pretend the last submitted task completes first. Preserve input order.
        values={m:observation(m) for m in reversed(masks)}
        return [values[m] for m in masks]
    ctx.native_workers,ctx.native_batch=2,ordered_batch
    new=run(ctx)
    assert sizes and max(sizes)==4 and len(set(all_masks))==len(all_masks)
    assert len(all_masks)+1==new[next(iter(new))].diagnostics["oracle_prefixes"]
    for m in old:
        np.testing.assert_array_equal(old[m].scores,new[m].scores)
        assert old[m].selected==new[m].selected and old[m].diagnostics==new[m].diagnostics

