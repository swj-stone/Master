"""Exact small-budget compression by conditioning worst exchange votes on the omitted set."""
import numpy as np
from numba import njit
from minppi_mofs.radius_minpi import incumbent


@njit(cache=True)
def _suffix_min(weights,k):
    n=len(weights)
    v=np.full((n+1,k+1),np.inf)
    v[:,0]=0.
    for j in range(n-1,-1,-1):
        for t in range(1,min(k,n-j)+1):
            v[j,t]=min(v[j+1,t],weights[j]+v[j+1,t-1])
    return v


@njit(cache=True)
def _search(g,d,a,z,use_vote,use_plane):
    k,q=len(a),len(z)
    # All computations below are on rounded integer coefficients.
    upper_plane=np.zeros((k+1,q))
    worst=np.zeros((k+1,k,q))
    for b,j in enumerate(z):
        gp=np.empty(q-1)
        for c,l in enumerate(z):
            if c!=b:
                gp[c-(c>b)]=g[j,l]
        gp.sort()
        for r in range(2,k+1):
            upper_plane[r,b]=upper_plane[r-1,b]+gp[r-2]
        if use_vote:
            for s,i in enumerate(a):
                diffs=np.empty(q-1)
                for c,l in enumerate(z):
                    if c!=b:
                        diffs[c-(c>b)]=g[i,l]-g[j,l]
                diffs.sort()
                for r in range(2,k+1):
                    worst[r,s,b]=worst[r-1,s,b]+diffs[q-r]
    anchor=0.
    for s,i in enumerate(a):
        anchor+=d[i]
        for j in a[:s]:
            anchor-=2*g[i,j]
    high=anchor
    best=a.copy()
    visited=solved=0
    for bitset in range(1,1<<k):
        removed=np.empty(k,np.int64)
        retained=np.empty(k,np.int64)
        r=h=0
        base=0.
        for s,i in enumerate(a):
            if bitset & (1<<s):
                removed[r]=s
                r+=1
            else:
                for m in retained[:h]:
                    base-=2*g[i,m]
                base+=d[i]
                retained[h]=i
                h+=1
        if r==1:
            continue  # incumbent is a one-swap optimum
        linear=np.empty(q)
        plane=np.empty(q)
        vote=np.zeros(q)
        for b,j in enumerate(z):
            linear[b]=d[j]
            for l in retained[:h]:
                linear[b]-=2*g[j,l]
            # Store r times each lower-bound contribution, preserving exact integers.
            plane[b]=anchor-base-r*(linear[b]-upper_plane[r,b])
            if use_vote:
                for s in removed[:r]:
                    i=a[s]
                    gap=d[i]-d[j]
                    observed=gap
                    for l in a:
                        if l!=i:
                            observed-=2*(g[i,l]-g[j,l])
                    adversary=gap-2*worst[r,s,b]
                    for l in retained[:h]:
                        adversary-=2*(g[i,l]-g[j,l])
                    vote[b]+=observed+adversary
        # The lower bounds share the incumbent; they need not order outsiders alike.
        threshold=r*(anchor-high)
        if (use_plane and np.sort(plane)[:r].sum()>=threshold) or (
                use_vote and np.sort(vote)[:r].sum()>=2*threshold):
            solved+=1
            continue
        order=np.argsort(vote if use_vote and not use_plane else plane)
        bplane=_suffix_min(plane[order],r)
        bvote=_suffix_min(vote[order],r) if use_vote else bplane
        level=col=0
        selected=np.empty(r,np.int64)
        partial=np.zeros(r+1)
        partial[0]=base
        ps=np.zeros(r+1)
        vs=np.zeros(r+1)
        while level>=0:
            visited+=1
            t=r-level
            threshold=r*(anchor-high)
            if (col>q-t or (use_plane and ps[level]+bplane[col,t]>=threshold)
                    or (use_vote and vs[level]+bvote[col,t]>=2*threshold)):
                level-=1
                if level>=0:
                    col=selected[level]+1
                continue
            b=order[col]
            val=partial[level]+linear[b]
            for u in selected[:level]:
                val-=2*g[z[b],z[order[u]]]
            selected[level]=col
            if level==r-1:
                if val>high:
                    high=val
                    best[:h]=retained[:h]
                    for u in range(r):
                        best[h+u]=z[order[selected[u]]]
            else:
                ps[level+1]=ps[level]+plane[b]
                vs[level+1]=vs[level]+vote[b]
                partial[level+1]=val
                level+=1
            col+=1
    return best,visited,solved


def sector_best(g,d,k,method="both"):
    if method not in ("both","plane","vote"):
        raise ValueError(method)
    p=len(d)
    if not 0<=k<=p or g.shape!=(p,p):
        raise ValueError("Expected a square game and a feasible width")
    complement=k>p//2
    if complement:
        d=2*(g.sum(axis=1)-g.diagonal())-d
        k=p-k
    if k>8:
        raise ValueError("Exact sector search is intended for at most eight inclusions or omissions")
    mask=incumbent(g,d,k)
    a,z=np.flatnonzero(mask),np.flatnonzero(~mask)
    if k<2:
        chosen,visited,cut=a,0,0
    else:
        chosen,visited,cut=_search(g,d,a,z,method!="plane",method!="vote")
    if complement:
        chosen=np.setdiff1d(np.arange(p),chosen)
    return np.sort(chosen),dict(nodes=visited,sectors_excluded=cut)
