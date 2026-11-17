"""Exhaustively audit outage fidelity and its incremental solver on small path games."""
from itertools import combinations
import numpy as np
from scipy.sparse import csc_matrix,csr_matrix
from minppi_mofs.availability_minpi import AvailabilityGame


def main():
    r=np.random.default_rng(617)
    cnt=paths=0
    for p in range(1,8):
        hs=[h for depth in range(1,min(p,3)+1) for h in combinations(range(p),depth)]
        hi=np.array([sum(1<<j for j in h) for h in hs])
        inc=csr_matrix([[(j in h) for j in range(p)] for h in hs],dtype=float)
        masks=np.arange(1<<p)
        sets=np.array([[(m>>j)&1 for j in range(p)] for m in masks],bool)
        for trial in range(7):
            a=r.integers(-3,4,size=(5,len(hs))).astype(float)
            if trial==1: a*=0.
            if trial==2: a[:,:p]=0.
            if trial==3: a[0]=a[1]
            y=np.array([a[:,(m&hi)==hi].sum(axis=1) for m in masks])
            g=AvailabilityGame.from_paths(csc_matrix(a),inc)
            for m,S in zip(masks,sets):
                errors=np.sum((y-y[masks&m])**2,axis=1)
                curve=[np.mean(errors[sets.sum(axis=1)==size]) for size in range(p+1)]
                np.testing.assert_allclose(curve,g.conditional(g.coefficients(S)),atol=1e-10)
                cnt+=p+1
            for rule in ("intact","pooled","min"):
                for mode in (None,r.permutation(p)):
                    for rho in (0,.2,.6,1.):
                        ref,rdiag=g.search(rho,rule=rule,order=mode,algorithm="recount")
                        found,d=g.search(rho,rule=rule,order=mode)
                        np.testing.assert_array_equal(ref,found)
                        assert rdiag["path"]==d["path"]
                        if rule=="min":
                            assert np.all(d["physical_risk"]<=rho*rho*d["reference_energy"]+1e-8)
                        paths+=1
    print(f"{cnt} exact size-specific expected errors; {paths} identical complete paths")


if __name__=="__main__":
    main()
