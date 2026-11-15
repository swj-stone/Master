"""Finite-population test of context votes on fixed Walsh-polynomial predictors."""
import csv
import gzip
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.context_diagnostics.design import CONFIG, OUTPUT, ROOT, config, decide, digest, families
from minppi_mofs.experiment_protocol import atomic_write_json
from minppi_mofs.piv3_compact_witness import _sample


def run():
    spec = config()["synthetic"]
    anchor = frozenset(spec["anchor"])
    tail = spec["tail"]
    contexts = [frozenset(m) for size in range(len(tail)+1) for m in combinations(tail, size)]
    population = np.array(list(product((-1, 1), repeat=len(anchor)+len(tail)+1)), dtype=float)
    rows = []
    for seed in spec["seeds"]:
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(population), 256, replace=True)
        _, weights = _sample(np.zeros(256), "regression", np.random.SeedSequence([seed, 659]))
        rules = families(tail, len(anchor), np.random.default_rng(np.random.SeedSequence([seed, 67113])))
        for good,bad in permutations(tail, 2):
            terms = [frozenset((0,)),frozenset((1,)),frozenset((2,)),
                     frozenset((0,good)),frozenset((0,bad))]
            design = np.column_stack([population[:,list(t)].prod(axis=1) for t in terms])
            for scenario in spec["models"]:
                suitable = scenario in ("correct_interaction","opposed_interactions")
                spurious = scenario in ("spurious_interaction","opposed_interactions")
                delta = spec["interaction_amplitude"]
                truth = np.array([*spec["main_effects"],delta*suitable,0.0])
                model = np.array([*spec["main_effects"],delta*suitable,delta*spurious])
                y = design @ truth + population[:,-1]
                cache = {}
                def risk(mask):
                    observed = tuple(t.issubset(mask) for t in terms)
                    if observed not in cache:
                        fitted = model*np.array(observed)
                        rm = 1.0 + np.square(truth-fitted).sum()
                        losses = np.square(y - design @ fitted)
                        if not np.isclose(rm, losses.mean(), atol=1e-12,rtol=0):
                            raise ArithmeticError("Analytic population risk disagrees with enumeration")
                        cache[observed] = rm, weights @ losses[sample]
                    return cache[observed]
                for dropped in sorted(anchor):
                    support = anchor-{dropped}
                    exact, observed = {}, {}
                    for e in contexts:
                        baseline, panels = risk(anchor | e)
                        reduced, cp = risk(support | e)
                        exact[e] = 4*int(reduced <= 1.03*baseline+1e-12)
                        observed[e] = int((cp <= 1.03*panels+1e-12).sum())
                    oracle_failures = sum(n < 3 for n in exact.values())
                    for setting, votes in (("population",exact),("sample",observed)):
                        for rule, ext in rules.items():
                            accept, visits = decide(rule,[votes[e] for e in ext])
                            rows.append(dict(scenario=scenario,seed=seed,useful=good,spurious=bad,
                                             dropped=dropped,evaluation=setting,rule=rule,
                                             context_count=len(ext),visits=visits,accept=int(accept),
                                             exhaustive_violations=oracle_failures))
    out = OUTPUT/"synthetic"
    out.mkdir(parents=True,exist_ok=True)
    with gzip.open(out/"decisions.csv.gz","wt",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import pandas as pd
    data=pd.DataFrame(rows)
    data["false_accept"]=data.accept.astype(bool) & data.exhaustive_violations.gt(0)
    data["false_reject"]=~data.accept.astype(bool) & data.exhaustive_violations.eq(0)
    grouped=data.groupby(["dropped","scenario","evaluation","rule"],sort=False)
    grouped.agg(n=("accept","size"),accepted=("accept","sum"),false_accept=("false_accept","sum"),
                false_reject=("false_reject","sum"),mean_visits=("visits","mean")).reset_index().to_csv(
        out/"summary.csv",index=False)
    sources=[Path(__file__),Path(__file__).with_name("design.py"),CONFIG,
             ROOT/"src/minppi_mofs/context_design.py",ROOT/"src/minppi_mofs/piv3_compact_witness.py"]
    atomic_write_json(out/"pins.json",{p.relative_to(ROOT).as_posix():digest(p) for p in sources})
    print(f"{len(rows)} deterministic proposed-deletion decisions exported to {out}",flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
