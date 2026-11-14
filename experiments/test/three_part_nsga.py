"""Gain-seeded binary NSGA-II for a budget of distinct feature subsets."""
from itertools import combinations
from math import ceil
from numbers import Integral

import numpy as np

from minppi_mofs.baselines.nsga2_fs import _compute_hv, _crowding_distance, _non_dominated_sort


REF_POINT = (1.1, 1.1)


def _initial_candidates(width, ordered):
    yield tuple(range(width)), "full_support"
    # Span sparse to wide nested supports; the number of seed prefixes does
    # not grow with the number of columns. Every arm receives the same rule.
    for k in (1, ceil(width**.25), round(np.cbrt(width)), ceil(np.sqrt(width)), round(width**(2/3))):
        yield tuple(sorted(ordered[:k])), "gain_prefix"


def _random_mask(rng, width):
    # Log-uniform widths allow a random immigrant to be sparse in a wide table.
    k = int(np.clip(np.rint(np.exp(rng.uniform(0.,np.log(width)))),1,width))
    return tuple(sorted(rng.choice(width,k,replace=False).tolist()))


def _fronts(pool, results):
    objectives = np.array([results[i]["objectives"] for i in pool])
    fronts = _non_dominated_sort(objectives)
    rank = np.empty(len(pool),dtype=int)
    crowding = np.empty(len(pool))
    for level, front in enumerate(fronts):
        rank[front] = level
        crowding[front] = _crowding_distance(objectives,front)
    return fronts, rank, crowding


def _tournament(rng, pool, ranks, crowding):
    left, right = rng.choice(len(pool), 2, replace=False)
    if (ranks[left],-crowding[left]) == (ranks[right],-crowding[right]):
        return pool[rng.choice([left,right])]
    return pool[min((left,right),key=lambda i: (ranks[i],-crowding[i]))]


def _child(rng, width, first, second):
    a = np.zeros(width,dtype=bool)
    b = np.zeros(width,dtype=bool)
    a[list(first)] = True
    b[list(second)] = True
    child = np.where(rng.random(width) < .5,a,b)
    child ^= rng.random(width) < 1./width
    if not child.any():
        child[rng.integers(width)] = True
    return tuple(np.flatnonzero(child).tolist())


def search(support, columns, gain_order, evaluate, progress, *, seed=42, population=8, budget=24):
    """Minimize input-relative width and primary rank-side prediction loss.

    Refits and selection of a deployment mask belong to the caller. Every distinct
    nonempty mask is evaluated exactly once. The archive contains every evaluated
    nondominated alternative, including ties with different retained features.
    """
    columns, support, gain_order = list(columns), list(support), list(gain_order)
    all_names = set(columns)
    allowed = set(support)
    if (len(columns) != len(all_names) or not allowed or len(support) != len(allowed)
            or not allowed <= all_names or not set(gain_order) <= all_names
            or len(gain_order) != len(set(gain_order))):
        raise ValueError("Expected unique feature names, a nonempty support and a valid Gain order")
    if (not isinstance(population,Integral) or isinstance(population,bool) or population < 2
            or not isinstance(budget,Integral) or isinstance(budget,bool) or budget < 1):
        raise ValueError("Population must be at least two and the budget positive")
    names = [col for col in columns if col in allowed]
    width = len(names)
    indices = {name:i for i,name in enumerate(names)}
    # The caller may supply only split-used coordinates. Preserve unused ones as
    # an unranked tail, in original order, so the plain search can reach every input.
    present = set(gain_order)
    ordered = [indices[x] for x in gain_order if x in allowed] + [i for i,x in enumerate(names) if x not in present]
    domain_size = (1 << width) - 1 if width <= int(budget).bit_length() else None
    cap = min(budget,domain_size) if domain_size is not None else budget
    exhaustive = domain_size == cap
    rng = np.random.default_rng(seed)
    results, masks, seen, hv_history = [], [], set(), []

    def add(mask, origin, gen):
        if mask in seen:
            return None
        cols = [names[j] for j in mask]
        if progress is not None:
            progress("nsga_search", checked=len(results),total=cap,k=len(cols),generation=gen)
        scores = {key:float(value) for key,value in evaluate(cols).items()}
        primary = scores.get("auc", scores.get("r2"))
        if primary is None or not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("Expected a finite AUC or R² and finite secondary metrics")
        idx = len(results)
        obj = [len(cols)/len(columns),1.-float(np.clip(primary,0,1))]
        results.append(dict(id=idx,names=cols,k=len(cols),scores=scores,objectives=obj,generation=gen,origin=origin))
        seen.add(mask)
        masks.append(mask)
        return idx

    def archive_hv():
        return _compute_hv(np.array([r["objectives"] for r in results]),np.asarray(REF_POINT))

    if exhaustive:
        add(tuple(range(width)),"full_support",0)
        for k in range(1,width):
            for mask in combinations(range(width),k):
                add(mask,"exhaustive_small_space",0)
        hv_history.append(archive_hv())
    else:
        size = min(population,cap)
        for mask,origin in _initial_candidates(width,ordered):
            if len(results) == size:
                break
            add(mask,origin,0)

        # For tiny spaces near exhaustion, repeated random proposals can collide.
        # Deterministic fallback probes at most len(seen)+1 distinct nonempty masks.
        def immigrant():
            for _ in range(24):
                mask = _random_mask(rng,width)
                if mask not in seen:
                    return mask
            for code in range(1,len(seen)+2):
                mask = tuple(j for j in range(code.bit_length()) if code & (1<<j))
                if mask not in seen:
                    return mask
            raise AssertionError("Expected an unseen nonempty mask")

        while len(results) < size:
            add(immigrant(),"initial_random",0)
        population_ids = list(range(size))
        hv_history.append(archive_hv())
        gen = 0
        while len(results) < cap:
            gen += 1
            _,ranks,crowding = _fronts(population_ids, results)
            offspring = []
            for _ in range(min(size,cap-len(results))):
                origin = "variation"
                for _ in range(24):
                    first = _tournament(rng,population_ids,ranks,crowding)
                    second = _tournament(rng,population_ids,ranks,crowding)
                    mask = _child(rng,width,masks[first],masks[second])
                    if mask not in seen:
                        break
                else:
                    mask = immigrant()
                    origin = "immigrant"
                offspring.append(add(mask,origin,gen))
            combined = population_ids + offspring
            fronts, _, crowding = _fronts(combined,results)
            population_ids = []
            for front in fronts:
                remaining = size-len(population_ids)
                if len(front) > remaining:
                    # Shuffle before the stable sort to break exact crowding ties symmetrically.
                    front = rng.permutation(front)
                    front = front[np.argsort(-crowding[front],kind="stable")[:remaining]]
                population_ids.extend(combined[i] for i in front)
                if len(population_ids) == size:
                    break
            hv_history.append(archive_hv())

    all_objectives = np.array([r["objectives"] for r in results])
    front = [results[i] for i in _non_dominated_sort(all_objectives)[0]]
    front.sort(key=lambda r:(r["k"],r["objectives"][1],r["id"]))
    return dict(front=front,evaluations=results,hv=hv_history[-1],hv_history=hv_history,
                unique_evaluations=len(results),budget=int(budget),population=int(population),
                exhaustive=exhaustive,ref_point=list(REF_POINT),original_width=len(columns),
                support_width=width,objective_names=["k/original_width","1-clipped_auc_or_r2"])

