"""Loss of the mean tree predictor at exact coalition sizes."""
import numpy as np
from numba import njit
from scipy import sparse
from experiments.minpi_rework.quota_compress_tree_native_exact import initialize


def contain(pop,k,d):
    result=np.ones(len(d))
    for h in range(d.max(initial=0)):
        if h>=pop:
            result[d>h]=0.
        else:
            result[d>h]*=max(0,k-h)/(pop-h)
    return result


def votes(g,bands=False):
    d=np.diff(g.incidence.indptr)
    out=[]
    sizes=np.array_split(np.arange(g.p),3) if bands else ([0],[(g.p-1)//2],[g.p-1])
    for sizes_s in sizes:
        w=np.mean([contain(g.p-1,s,d) for s in sizes_s],axis=0)
        extra=np.mean([contain(g.p-1,s,d-1) for s in sizes_s],axis=0)
        # Remove the groups involving j from the unconditional path sum.
        exclude=(g.a@g.incidence.multiply(w[:,None])).toarray()
        delta=(g.a@g.incidence.multiply(extra[:,None])).toarray()
        residual=g.full[:,None]-(g.a@w)[:,None]+exclude
        out.append(2*np.einsum("ij,ij->j",residual,delta)-np.einsum("ij,ij->j",delta,delta))
    return np.asarray(out)


def refined(g,points):
    bands=votes(g,True)
    index=np.arange(g.p)
    lim=1e-12*g.norm**2
    result={}
    for name,matrix in (("point",points),("band",bands)):
        # The numerical zero threshold is relative to total centered teacher energy.
        arr=matrix.copy()
        arr[np.abs(arr)<=lim]=0.
        vals=np.sort(arr,axis=0)
        result[name+"lex"]=np.lexsort((-index,-vals[2],-vals[1],-vals[0]))
        result[name+"harm"]=np.lexsort((-index,-points[-1],vals[0]<0))
        if name=="band":
            result["bandmin"]=np.lexsort((-index,-vals[0]))
            result["bandpool"]=np.lexsort((-index,-arr.mean(axis=0)))
    return result


def panels(g):
    q=np.ceil(g.p*np.array([.25,.5,1.])).astype(int)
    degree=np.diff(g.incidence.indptr)
    w=np.array([contain(g.p,k,degree) for k in q])
    norm=np.array([np.linalg.norm(g.a@row) for row in w])
    # Exact zero means there is zero centered variation at this size; the toy
    # constant game also has zero subgroup increments. Training data are positive.
    if np.any(norm<=0):
        raise ValueError("A coalition mean is constant; normalized guard is undefined")
    a=sparse.vstack([g.a.multiply(row/scale) for row,scale in zip(w,norm)],format="csc")
    return q,w,norm,a


@njit(cache=True)
def solve_kernel(n,ap,ai,av,ip,ii,gp,gi,order,greedy,mode):
    p=len(order)
    h=3
    d=initialize(n*h,ap,ai,av,ip,ii,p)
    r=np.zeros(n*h)
    keep=np.ones(p,np.bool_)
    active=np.ones(len(ap)-1,np.bool_)
    checks=0
    for _ in range(p):
        before=np.sum(keep)
        best=.04+1e-12
        winner=-1
        for ptr in range(p-1,-1,-1):
            j=order[ptr]
            if not keep[j]:
                continue
            risk=0.
            checks+=1
            for s in range(2,-1,-1):
                loss=0.
                for i in range(s*n,(s+1)*n):
                    u=r[i]+d[j,i]
                    loss+=u*u
                risk=risk+loss/h if mode==1 else max(risk,loss)
                if mode==2 or risk>best:
                    break
            if risk<best or (risk==best and winner<0):
                winner=j
                if greedy:
                    best=risk
                else:
                    drop(j,d,r,keep,active,ap,ai,av,ip,ii,gp,gi)
        if greedy and winner>=0:
            drop(winner,d,r,keep,active,ap,ai,av,ip,ii,gp,gi)
        if np.sum(keep)==before:
            break
    errors=np.array([np.sqrt(np.sum(r[s*n:(s+1)*n]**2)) for s in range(h)])
    return np.where(keep)[0],errors,checks


@njit(cache=True)
def drop(j,d,r,keep,active,ap,ai,av,ip,ii,gp,gi):
    r+=d[j]
    keep[j]=False
    for z in range(gp[j],gp[j+1]):
        group=gi[z]
        if active[group]:
            active[group]=False
            for a in range(ip[group],ip[group+1]):
                h=ii[a]
                if keep[h]:
                    for pos in range(ap[group],ap[group+1]):
                        d[h,ai[pos]]-=av[pos]


def solve(g,a,order,greedy,mode):
    inc=g.incidence
    by=inc.tocsc()
    return solve_kernel(len(g.full),a.indptr,a.indices,a.data,inc.indptr,inc.indices,by.indptr,by.indices,
        order,greedy,{"max":0,"pool":1,"full":2}[mode])


@njit(cache=True)
def expected(x,left,right,feature,cut,default,value,heads,group,lookup,weights,c):
    out=np.zeros((len(x),c,len(weights)))
    for i in range(len(x)):
        for t in range(len(heads)):
            node=0
            while left[t,node]>=0:
                j=lookup[feature[t,node]]
                if j<0:
                    break
                v=x[i,j]
                go=default[t,node] if np.isnan(v) else v<cut[t,node]
                child=left[t,node] if go else right[t,node]
                delta=value[t,child]-value[t,node]
                g=group[t,child]
                for s in range(len(weights)):
                    out[i,heads[t],s]+=delta*weights[s,g]
                node=child
    return out


def predict(f,x,cols,w):
    lookup=np.full(f.p,-1,int)
    lookup[cols]=np.arange(len(cols))
    a=expected(np.ascontiguousarray(x[:,cols],np.float32),*f.buffers(),f.group,lookup,w,f.c)
    return a+f.empty[None,:,None]


def check():
    from itertools import combinations
    from experiments.minpi_rework.quota_compress_tree import Game
    rng=np.random.default_rng(346)
    p=8
    sets=[(j,) for j in range(p)]+list(combinations(range(p),2))[::3]+[(0,1,4),(2,5,6)]
    inc=sparse.csr_matrix([[float(j in s) for j in range(p)] for s in sets])
    g=Game(sparse.csc_matrix(rng.normal(size=(15,len(sets)))),inc)
    value=lambda c: np.asarray(g.a[:,[set(s)<=c for s in sets]].sum(axis=1)).ravel()
    actual=votes(g)
    for j in range(p):
        for t,s in enumerate((0,(p-1)//2,p-1)):
            cs=list(combinations(set(range(p))-{j},s))
            before=np.mean([value(set(c)) for c in cs],axis=0)
            after=np.mean([value(set(c)|{j}) for c in cs],axis=0)
            np.testing.assert_allclose(actual[t,j],np.sum((g.full-before)**2)-np.sum((g.full-after)**2))
    q,w,_,a=panels(g)
    for i,k in enumerate(q):
        np.testing.assert_allclose(g.a@w[i],np.mean([value(set(c)) for c in combinations(range(p),k)],axis=0))
    order=np.arange(p)
    cols,_,_=solve(g,a,order,True,"full")
    np.testing.assert_array_equal(cols,g.search("exact_greedy",order)[0])
