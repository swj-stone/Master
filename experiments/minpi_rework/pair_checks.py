"""Compare every initial block marginal with exhaustive coalition games."""
from itertools import combinations

import numpy as np

from minppi_mofs.additive_minpi import quadratic
from minppi_mofs.pair_vote_path import block_scores,pair_order
from experiments.minpi_rework.pair_outage import exchange_half,outage_order,risks,support_risk


def check():
    rng=np.random.default_rng(119)
    for p in (1,2,6):
        parts=rng.normal(size=(13,p,2))
        target=rng.normal(size=(13,2))
        gram,single=quadratic(parts,target)
        def utility(cols):
            residual=target-parts[:,cols].sum(axis=1)
            return -np.sum(residual**2)/len(target)
        def outage(active,rule):
            vals=np.array([-utility(active),*[-utility([j for j in active if j!=gone]) for gone in active]])
            return {"min":vals.max(),"mean":vals.mean(),"point":vals[0]}[rule]
        order=rng.permutation(p)
        actual,worst=risks(gram,single,-utility([]),order)
        for k in range(1,p+1):
            cols=order[:k].tolist()
            np.testing.assert_allclose([actual[k-1],worst[k-1]],[-utility(cols),outage(cols,"min")])
            np.testing.assert_allclose(support_risk(gram,single,-utility([]),cols),outage(cols,"min"))
        for rule in ("min","mean","point"):
            half=(p+1)//2
            initial=sorted(np.argsort(-gram.diagonal(),kind="stable")[:half])
            choices=[outage(initial,rule)]
            for gone in initial:
                for new in set(range(p))-set(initial):
                    choices.append(outage([j for j in initial if j!=gone]+[new],rule))
            one=exchange_half(gram,single,rule,rounds=1)[:half].tolist()
            np.testing.assert_allclose(outage(one,rule),min(choices),atol=1e-12)
            order=outage_order(gram,single,rule,False)
            for k in range(p,1,-1):
                cols=order[:k].tolist()
                expected=min(outage([j for j in cols if j!=skip],rule) for skip in cols)
                np.testing.assert_allclose(outage(cols[:-1],rule),expected,atol=1e-12)
        for rule in ("min","pooled","point","drop_min"):
            l,r,score,_=block_scores(gram,single,rule)
            for a,b,s in zip(l,r,score,strict=True):
                block=sorted(set([a,b]))
                other=sorted(set(range(p))-set(block))
                diff=[utility(list(k)+block)-utility(list(k)) for h in range(len(other)+1) for k in combinations(other,h)]
                expected={"min":min(diff),"pooled":np.mean(diff),"point":diff[-1],"drop_min":max(diff)}[rule]
                np.testing.assert_allclose(s*len(block),expected,atol=1e-12)
            order,blocks=pair_order(gram,single,rule)
            assert sorted(order)==list(range(p))
            if rule=="min":
                # For arbitrary set utilities, the minimum block addition telescopes
                # to individual additions, all bounded below by their own minima.
                assert all(len(k)==1 for k in blocks)
            active=list(range(p))
            for block in blocks:
                cols=np.array(active)
                l,r,s,_=block_scores(gram[np.ix_(cols,cols)],single[cols],rule)
                actual=next(v for a,b,v in zip(l,r,s,strict=True) if set([active[a],active[b]])==set(block))
                np.testing.assert_allclose(actual,min(s),atol=1e-12)
                active=[i for i in active if i not in block]
    print("Exact single/pair votes and every active-stage update agree with enumeration")


if __name__=="__main__":
    check()
