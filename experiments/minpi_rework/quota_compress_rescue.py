"""Bounded exact add-back and pair deletion for frozen tree path increments."""
import numpy as np
from numba import njit
from scipy import sparse


def ballot_matrix(game):
    a=game.a
    p=game.p
    degrees=np.diff(game.incidence.indptr)
    weight=np.ones((a.shape[1],3))
    for s,k in enumerate((max(1,p//4),max(1,p//2),p)):
        for h in range(degrees.max(initial=0)):
            weight[degrees>h,s]*=max(0,k-h)/(p-h)
    expected=a@weight
    scale=np.linalg.norm(expected,axis=0)
    expected/=np.where(scale>0,scale,1.)
    return sparse.vstack((a,sparse.csr_matrix((a.T@expected).T)),format="csc")


@njit(cache=True)
def _loss(x,y,n,limit,slack,ballot,counts):
    if ballot:
        for s in range(n,len(x)):
            z=x[s]+y[s]
            if z*z>limit+slack:
                counts[0]+=1
                return np.inf
    total=0.
    counts[1]+=1
    for start in range(0,n,32):
        for i in range(start,min(n,start+32)):
            z=x[i]+y[i]
            total+=z*z
        if total>limit:
            return np.inf
    return total


@njit(cache=True)
def _state(ap,ai,av,ip,ii,keep,length):
    residual=np.zeros(length)
    d=np.zeros((len(keep),length))
    restore=np.zeros_like(d)
    missing=np.empty(len(ap)-1,np.int32)
    for g in range(len(missing)):
        absent=-1
        for pos in range(ip[g],ip[g+1]):
            j=ii[pos]
            if not keep[j]:
                if absent>=0:
                    absent=-2
                    break
                absent=j
        missing[g]=absent
        for z in range(ap[g],ap[g+1]):
            i=ai[z]
            v=av[z]
            if absent==-1:
                for pos in range(ip[g],ip[g+1]):
                    d[ii[pos],i]+=v
            else:
                residual[i]+=v
                if absent>=0:
                    restore[absent,i]+=v
    return residual,d,restore,missing


@njit(cache=True)
def _overlap(j,a,sign,missing,d,ap,ai,av,ip,ii,gp,gi):
    for slot in range(gp[j],gp[j+1]):
        g=gi[slot]
        if missing[g]==-1 or missing[g]==a:
            for pos in range(ip[g],ip[g+1]):
                h=ii[pos]
                if h==j:
                    continue
                for z in range(ap[g],ap[g+1]):
                    d[h,ai[z]]+=sign*av[z]


@njit(cache=True)
def core(n,length,ap,ai,av,ip,ii,gp,gi,initial,limit,pair,ballot,rounds,slots,firsts):
    keep=initial.copy()
    scale=max(1e-100,limit/.04)
    slack=2e-12*max(1.,scale)
    counts=np.zeros(5,np.int64)   # directional rejections, norm checks, pairs, swaps, singleton repairs
    moves=np.full((rounds,3),-1,np.int32)
    completed=0
    p=len(keep)
    finish=0
    while True:
        r,base,restore,missing=_state(ap,ai,av,ip,ii,keep,length)
        current=np.sum(r[:n]*r[:n])
        if completed>0:
            best=limit
            chosen=-1
            for j in range(p):
                if keep[j]:
                    risk=_loss(r,base[j],n,best,slack,ballot,counts)
                    if risk<best or (risk==best and chosen<0):
                        chosen=j
                        best=risk
            if chosen>=0:
                keep[chosen]=False
                counts[4]+=1
                continue
        if completed>=rounds or not np.any(keep):
            finish=1 if completed>=rounds else 2
            break
        restored_loss=np.full(p,np.inf)
        for a in range(p):
            if not keep[a]:
                trial=r[:n]-restore[a,:n]
                restored_loss[a]=np.sum(trial*trial)
        best1=current-1e-10*scale
        best2=limit
        add1=drop1=add2=drop2=drop3=-1
        restorations=np.argsort(restored_loss,kind="mergesort")
        for pos in range(min(slots,len(restorations))):
            a=restorations[pos]
            if keep[a]:
                break
            rr=r-restore[a]
            d=base.copy()
            for slot in range(gp[a],gp[a+1]):
                g=gi[slot]
                if missing[g]==a:
                    for u in range(ip[g],ip[g+1]):
                        j=ii[u]
                        if keep[j]:
                            for z in range(ap[g],ap[g+1]):
                                d[j,ai[z]]+=av[z]
            singles=np.full(p,np.inf)
            for j in range(p):
                if keep[j]:
                    # Exact ranks determine the deterministic top-four neighborhood.
                    singles[j]=_loss(rr,d[j],n,np.inf,slack,False,counts)
                    if singles[j]<best1:
                        best1=singles[j]
                        add1=a
                        drop1=j
            if not pair:
                continue
            first=np.argsort(singles,kind="mergesort")
            tested=np.zeros(p,np.bool_)
            for ptr in range(min(firsts,p)):
                j=first[ptr]
                if not keep[j]:
                    break
                tested[j]=True
                rj=rr+d[j]
                _overlap(j,a,-1.,missing,d,ap,ai,av,ip,ii,gp,gi)
                for h in range(p):
                    if keep[h] and not tested[h]:
                        counts[2]+=1
                        risk=_loss(rj,d[h],n,best2,slack,ballot,counts)
                        if risk<best2 or (risk==best2 and add2<0):
                            best2=risk
                            add2=a
                            drop2=j
                            drop3=h
                _overlap(j,a,1.,missing,d,ap,ai,av,ip,ii,gp,gi)
        if add2>=0:
            keep[add2]=True
            keep[drop2]=False
            keep[drop3]=False
            moves[completed,0]=add2
            moves[completed,1]=drop2
            moves[completed,2]=drop3
        elif add1>=0:
            keep[add1]=True
            keep[drop1]=False
            counts[3]+=1
            moves[completed,0]=add1
            moves[completed,1]=drop1
        else:
            break
        completed+=1
    return np.where(keep)[0],moves[:completed],counts,finish


def search(game,initial,*,pair=True,ballot=False,rounds=4,slots=8,firsts=4):
    a=ballot_matrix(game) if ballot else game.a
    inc=game.incidence
    by=inc.tocsc()
    keep=np.zeros(game.p,bool)
    keep[initial]=True
    limit=(.2*game.norm+1e-12*max(1.,game.norm))**2
    return core(len(game.full),a.shape[0],a.indptr,a.indices,a.data,inc.indptr,inc.indices,
                by.indptr,by.indices,keep,limit,pair,ballot,rounds,slots,firsts)
