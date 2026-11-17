"""Enumerate small teacher-fidelity games to check signed fixings and exact certificates."""
from itertools import product

import numpy as np

from experiments.minpi_rework import quota_compress as q


def run():
    rng=np.random.default_rng(9127)
    ballots_checked=0
    solved=0
    for p in range(2,10):
        for case in range(3):
            a=rng.normal(size=(p+1,p))
            if case==1:
                a[:,1]=a[:,0]
            if case==2:
                a[:,1]=-a[:,0]
            g=a.T@a/max(1e-20,(a.sum(axis=1)**2).sum())
            rho=(.2,.4,.65)[case]
            choices=np.array(list(product((False,True),repeat=p)))
            errors=np.einsum("ni,ij,nj->n",~choices,g,~choices)
            exact=choices[errors<=rho**2+1e-12]
            opt=int(exact.sum(axis=1).min())
            incumbent,path=q.greedy(g,rho)
            w,quotas=q.initial_ballots(g,path,rho)
            assert q.lower_bound(w,quotas)<=opt
            for k in range(opt,p+1):
                possible=exact[exact.sum(axis=1)<=k]
                signs=q.fix(w,quotas,k)
                assert possible[:,signs==1].all() and not possible[:,signs==-1].any()
                ballots_checked+=1
            for m in (False,True):
                support,info=q.master(g,incumbent,path,m,rho,seconds=4.)
                assert info["lb"]<=opt<=support.sum() and q.feasible(g,support,rho)
                assert info["certified"] and info["lb"]==opt
                solved+=1
            support,info=q.cone(g,incumbent,rho,seconds=4.)
            assert int(support.sum())==info["lb"]==opt or not info["certified"]
    print(f"Verified {ballots_checked} conditional partitions and {solved} master optimum certificates.",flush=True)


if __name__=="__main__":
    run()
