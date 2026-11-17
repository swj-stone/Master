"""Bounded scalar projection of a truncated ensemble onto full centered teacher margins."""
import numpy as np
from numba import njit
from experiments.minpi_rework.quota_compress_rescue import ballot_matrix
from experiments.minpi_rework.quota_compress_tree_native_exact import initialize


@njit(cache=True)
def formula(ff,hh,fh,scaled):
    lam=min(2.,max(.5,fh/hh)) if scaled and hh>0. else 1.
    return max(0.,ff+lam*(lam*hh-2*fh)),lam


@njit(cache=True)
def evaluate(target,current,delta,n,best,tol,scaled,partial,votes,counts):
    if votes:
        lo=.5 if scaled else 1.
        hi=2. if scaled else 1.
        radius=np.sqrt(best+tol)
        for j in range(n,len(current)):
            h=current[j]-delta[j]
            z=target[j]
            if h==0.:
                if abs(z)>radius:
                    counts[1]+=1
                    return np.inf
            else:
                a=(z-radius)/h
                b=(z+radius)/h
                lo=max(lo,min(a,b))
                hi=min(hi,max(a,b))
                if lo>hi:
                    counts[1]+=1
                    return np.inf
    counts[0]+=1
    ff=hh=fh=0.
    for s in range(0,n,64):
        for i in range(s,min(n,s+64)):
            f=target[i]
            h=current[i]-delta[i]
            ff+=f*f
            hh+=h*h
            fh+=f*h
        if partial and formula(ff,hh,fh,scaled)[0]>best+tol:
            counts[2]+=1
            return np.inf
    return formula(ff,hh,fh,scaled)[0]


@njit(cache=True)
def kernel(n,ap,ai,av,ip,ii,gp,gi,order,target,threshold,scaled,greedy,partial,votes):
    p=len(order)
    d=initialize(len(target),ap,ai,av,ip,ii,p)
    current=target.copy()
    alive=np.ones(len(ap)-1,np.bool_)
    keep=np.ones(p,np.bool_)
    path=np.empty(p,np.int32)
    counts=np.zeros(3,np.int64)
    t=0
    tol=2e-12*max(1.,threshold/.04)
    for _ in range(p):
        best=threshold
        chosen=-1
        before=t
        for pos in range(p-1,-1,-1):
            j=order[pos]
            if not keep[j]:
                continue
            val=evaluate(target,current,d[j],n,best,tol,scaled,partial,votes,counts)
            if val<best or (val==best and chosen<0):
                chosen=j
                if greedy:
                    best=val
                    continue
                current-=d[j]
                path[t]=j
                t+=1
                drop(j,d,keep,alive,ap,ai,av,ip,ii,gp,gi)
        if greedy and chosen>=0:
            current-=d[chosen]
            path[t]=chosen
            t+=1
            drop(chosen,d,keep,alive,ap,ai,av,ip,ii,gp,gi)
        if t==before:
            break
    f=target[:n]
    h=current[:n]
    risk,lam=formula(f@f,h@h,f@h,scaled)
    return np.where(keep)[0],path[:t],risk,lam,counts


@njit(cache=True)
def drop(j,d,keep,alive,ap,ai,av,ip,ii,gp,gi):
    keep[j]=False
    for ptr in range(gp[j],gp[j+1]):
        g=gi[ptr]
        if alive[g]:
            alive[g]=False
            for pos in range(ip[g],ip[g+1]):
                h=ii[pos]
                if keep[h]:
                    for z in range(ap[g],ap[g+1]):
                        d[h,ai[z]]-=av[z]


def search(game,order,scaled=True,greedy=True,partial=True,votes=False):
    a=ballot_matrix(game) if votes else game.a
    target=np.asarray(a.sum(axis=1)).ravel() if votes else game.full
    by=game.incidence.tocsc()
    inc=game.incidence
    return kernel(len(game.full),a.indptr,a.indices,a.data,inc.indptr,inc.indices,by.indptr,
        by.indices,order,target,(.2*game.norm+1e-12*max(1.,game.norm))**2,scaled,greedy,partial,votes)


def check():
    from scipy import sparse
    from experiments.minpi_rework.quota_compress_tree import Game
    rng=np.random.default_rng(441)
    for slope in (.1,.5,1.,1.6,2.,3.):
        h=rng.normal(size=70)
        f=slope*h
        risk,lam=formula(f@f,h@h,f@h,True)
        np.testing.assert_allclose(lam,np.clip(slope,.5,2.))
        np.testing.assert_allclose(risk,np.sum((f-lam*h)**2),atol=1e-11)
    for p in (4,6,11):
        n=91
        g=Game(sparse.csc_matrix(rng.normal(size=(n,p))*rng.lognormal(0.,1.,p)),sparse.eye(p,format="csr"))
        order=rng.permutation(p)
        for scaled in (True,False):
            for greedy in (True,False):
                keep,path,risk,lam,_=search(g,order,scaled,greedy,False,False)
                for ballots in (True,False):
                    yes,yp,yr,yl,_=search(g,order,scaled,greedy,True,ballots)
                    np.testing.assert_array_equal(path,yp)
                    np.testing.assert_array_equal(keep,yes)
                    np.testing.assert_allclose([risk,lam],[yr,yl])
                u=np.asarray(g.a[:,keep].sum(axis=1)).ravel()
                np.testing.assert_allclose(risk,np.sum((g.full-lam*u)**2),atol=1e-10)
