"""Independent integer and exhaustive checks for the small-budget sector search."""
from itertools import combinations, product

import numpy as np

from minppi_mofs.exchange_minpi import integer_game
from minppi_mofs.sector_minpi import _search, _suffix_min, sector_best


def objective(g,d,s):
    """Python arbitrary-precision integer reference; G's diagonal cancels."""
    return sum(int(d[i]) for i in s)-2*sum(int(g[i,j]) for i,j in combinations(s,2))


def brute(g,d,k):
    return max(objective(g,d,s) for s in combinations(range(len(d)),k))


def check_solver(g,d,k):
    answer=brute(g,d,k)
    for mode in ("both","plane","vote"):
        s,_=sector_best(g,d,k,mode)
        assert len(s)==k and list(s)==sorted(set(s)) and all(0<=i<len(d) for i in s)
        assert objective(g,d,s)==answer,(mode,k,g,d,s,answer)
    return answer


def local_anchor(g,d,anchor):
    """Find a one-swap optimum from an arbitrary anchor using full objective calls."""
    a=tuple(sorted(anchor))
    while 0<len(a)<len(d):
        best=objective(g,d,a)
        nxt=a
        for i in a:
            for j in set(range(len(d)))-set(a):
                s=tuple(sorted((set(a)-{i})|{j}))
                val=objective(g,d,s)
                if val>best: best,nxt=val,s
        if nxt==a: break
        a=nxt
    return a


def endpoint_dif(g,d,i,j,background):
    return objective(g,d,tuple(background)+(i,))-objective(g,d,tuple(background)+(j,))


def bounds(g,d,a):
    z=tuple(sorted(set(range(len(d)))-set(a)))
    anchor=objective(g,d,a)
    checked=0
    for r in range(2,min(len(a),len(z))+1):
        for dropped in combinations(a,r):
            retained=tuple(sorted(set(a)-set(dropped)))
            base=objective(g,d,retained)
            midpoint,plane={},{}
            for j in z:
                potential=tuple(l for l in z if l!=j)
                # Minimize one outsider's end vote by exhaustive backgrounds.
                midpoint[j]=sum(endpoint_dif(g,d,i,j,tuple(t for t in a if t!=i))+
                    min(endpoint_dif(g,d,i,j,retained+b) for b in combinations(potential,r-1))
                    for i in dropped)
                minrow=min(sum(int(g[j,l]) for l in b) for b in combinations(potential,r-1))
                marginal=objective(g,d,retained+(j,))-base
                plane[j]=anchor-base-r*(marginal-minrow)
            for added in combinations(z,r):
                final=retained+added
                gap=anchor-objective(g,d,final)
                sums=sum(endpoint_dif(g,d,i,j,tuple(t for t in a if t!=i))+
                         endpoint_dif(g,d,i,j,tuple(t for t in final if t!=j))
                         for i in dropped for j in added)
                assert sums==2*r*gap
                assert sum(midpoint[j] for j in added)<=2*r*gap
                assert sum(plane[j] for j in added)<=r*gap
                checked+=1
    return checked


def signed(rng,p):
    u=np.triu(rng.integers(-7,8,(p,p)),1)
    g=u+u.T
    np.fill_diagonal(g,rng.integers(-13,14,p))
    return g.astype(float),rng.integers(-12,13,p).astype(float)


def run():
    rng=np.random.default_rng(4229)
    widths=forced=alt=nonlocal_opt=sectors=0
    # Exhaust all labelled signed unit games with four columns, including all
    # optimum ties. Diagonal values are arbitrary and carry no game information.
    for coeff in product((-1,1),repeat=10):
        g=np.diag([4.,3.,-2.,1.])
        for (i,j),v in zip(combinations(range(4),2),coeff[4:],strict=True):
            g[i,j]=g[j,i]=v
        for k in range(5):
            check_solver(g,np.asarray(coeff[:4],float),k); widths+=1
    # Larger signed, PSD, packed PSD and exactly tied/collinear games.
    for it in range(92):
        p=int(rng.integers(5,12))
        g,d=signed(rng,p)
        if it%4 in (1,2):
            x=rng.integers(-3,4,(4,p))
            x[:,1]=x[:,0]
            g=(x.T@x).astype(float)
            if it%4==2:
                y=rng.normal(size=(19,p)); y[:,1]=y[:,0]
                q=y.T@y; h=2*q.sum(1)-q.diagonal()
                g,d,_=integer_game(q,h)
            else: d[1]=d[0]
        if it%4==3: g[:]=rng.integers(-1,2); d[:]=rng.integers(-1,2)
        for k in range(p+1):
            check_solver(g,d,k); widths+=1
        # Independently check complementation, including the additive constant.
        cd=np.array([2*sum(int(g[i,j]) for j in range(p) if j!=i)-int(d[i]) for i in range(p)])
        k=int(rng.integers(2,p//2+1))
        src=tuple(sorted(rng.choice(p,k,replace=False)))
        co=tuple(sorted(set(range(p))-set(src)))
        assert objective(g,d,src)==objective(g,d,tuple(range(p)))+objective(g,cd,co)
        a=local_anchor(g,d,src)
        outside=np.array(sorted(set(range(p))-set(a)),int)
        answer=brute(g,d,k)
        for v,f in ((True,True),(True,False),(False,True)):
            s,_,_=_search(g,d,np.array(a),outside,v,f)
            assert objective(g,d,s)==answer; alt+=1
        nonlocal_opt+=int(objective(g,d,a)<answer)
        if p<=9: sectors+=bounds(g,d,a)
        # Random fixed-in and fixed-out constraint; optimize the remaining game.
        perm=rng.permutation(p)
        locked=list(perm[:1]); free=perm[2:]
        kk=int(rng.integers(0,len(free)+1))
        conditional=d[free]-2*g[np.ix_(free,locked)].sum(axis=1)
        remaining=check_solver(g[np.ix_(free,free)],conditional,kk)
        expected=max(objective(g,d,locked+list(s)) for s in combinations(free,kk))
        assert remaining+objective(g,d,locked)==expected
        forced+=1
    g,d=signed(rng,16)
    # Max supported internal r, along with nearly maximal coefficients from the
    # documented packing. Scale is chosen before screening, as in the benchmark.
    g,d,_=integer_game(g,d)
    check_solver(g,d,8); widths+=1
    for q in range(1,12):
        seq=rng.integers(-80,81,q)
        lim=min(q,8)
        dp=_suffix_min(seq.astype(float),lim)
        for i in range(q+1):
            for t in range(lim+1):
                if t<=q-i:
                    assert dp[i,t]==min(sum(group) for group in combinations(seq[i:],t))
                else: assert np.isinf(dp[i,t])
    assert nonlocal_opt>0
    print(f"{widths} exhaustive width comparisons x 3 modes; {alt} alternate local-anchor searches; "
          f"{nonlocal_opt} strict escapes; {forced} conditional games; {sectors} endpoint/plane checks")


if __name__=="__main__":
    run()
