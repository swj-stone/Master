"""Tree-margin fidelity under truncated traversal and minimum directional slack."""
from dataclasses import dataclass
import json

import numpy as np
from numba import njit
from scipy import sparse
from scipy.special import expit,softmax


@njit(cache=True)
def _predict(x,left,right,feature,cut,nan_left,value,heads,columns,c):
    out=np.zeros((len(x),c))
    for n in range(len(x)):
        for t in range(len(heads)):
            node=0
            while left[t,node]>=0:
                j=columns[feature[t,node]]
                if j<0:
                    break
                v=x[n,j]
                use_left=nan_left[t,node] if np.isnan(v) else v<cut[t,node]
                node=left[t,node] if use_left else right[t,node]
            out[n,heads[t]]+=value[t,node]
    return out


@njit(cache=True)
def _edges(x,left,right,feature,cut,nan_left,value,heads,group,c,max_entries):
    rows=np.empty(max_entries,np.int32)
    groups=np.empty(max_entries,np.int32)
    inc=np.empty(max_entries,np.float64)
    s=0
    for n in range(len(x)):
        for t in range(len(heads)):
            node=0
            while left[t,node]>=0:
                v=x[n,feature[t,node]]
                use_left=nan_left[t,node] if np.isnan(v) else v<cut[t,node]
                child=left[t,node] if use_left else right[t,node]
                rows[s]=n*c+heads[t]
                groups[s]=group[t,child]
                inc[s]=value[t,child]-value[t,node]
                s+=1
                node=child
    return rows[:s],groups[:s],inc[:s]


@njit(cache=True)
def _prefix(x,left,right,feature,cut,nan_left,value,heads,ranks,c):
    p=len(ranks)
    by_width=np.zeros((len(x)*c,p+1))
    for n in range(len(x)):
        for t in range(len(heads)):
            node=0
            last=0
            while left[t,node]>=0:
                f=feature[t,node]
                last=max(last,ranks[f])
                v=x[n,f]
                use_left=nan_left[t,node] if np.isnan(v) else v<cut[t,node]
                child=left[t,node] if use_left else right[t,node]
                by_width[n*c+heads[t],last]+=value[t,child]-value[t,node]
                node=child
    return by_width


@dataclass
class Forest:
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    cut: np.ndarray
    nan_left: np.ndarray
    value: np.ndarray
    heads: np.ndarray
    group: np.ndarray
    sets: list
    depth_sum: int
    p: int
    c: int
    offset: np.ndarray

    def buffers(self):
        return self.left,self.right,self.feature,self.cut,self.nan_left,self.value,self.heads

    def predict(self,x,columns):
        if x.shape[1]!=len(columns):
            raise ValueError("Sparse traversal requires the retained input matrix")
        positions=np.full(self.p,-1,dtype=np.int64)
        positions[columns]=np.arange(len(columns))
        return self.offset+_predict(np.asarray(x,dtype=np.float32),*self.buffers(),positions,self.c)

    @property
    def empty(self):
        return self.offset+np.bincount(self.heads,weights=self.value[:,0],minlength=self.c)

    def cache(self,x):
        x=np.asarray(x,dtype=np.float32)
        i,j,v=_edges(x,*self.buffers(),self.group,self.c,len(x)*self.depth_sum)
        a=sparse.coo_matrix((v,(i,j)),shape=(len(x)*self.c,len(self.sets))).tocsc()
        a.eliminate_zeros()
        ids=[i for i,s in enumerate(self.sets) for _ in s]
        cols=[j for s in self.sets for j in s]
        incidence=sparse.csr_matrix((np.ones(len(ids)),(ids,cols)),shape=(len(self.sets),self.p))
        return Game(a,incidence)

    def gain_prefix(self,x,order,rho):
        ranks=np.argsort(order)+1
        inc=_prefix(np.asarray(x,dtype=np.float32),*self.buffers(),ranks,self.c)
        residual=inc.sum(axis=1,keepdims=True)-np.cumsum(inc,axis=1)
        squared=np.einsum("ij,ij->j",residual,residual)
        k=int(np.flatnonzero(squared<=rho*rho*squared[0]+1e-12*max(1.,squared[0]))[0])
        return np.sort(order[:k]),dict(checks=len(order)+1)


def extract(model,row):
    from xgboost import DMatrix
    packed=json.loads(model.get_booster().save_raw(raw_format="json"))
    ensemble=packed["learner"]["gradient_booster"]["model"]
    heads=np.asarray(ensemble["tree_info"],int)
    c=int(heads.max()+1)
    trees=ensemble["trees"]
    shape=(len(trees),max(len(t["left_children"]) for t in trees))
    left=np.full(shape,-1,np.int32)
    right=left.copy()
    feature=np.zeros(shape,np.int32)
    group=left.copy()
    cut=np.zeros(shape,np.float32)
    nan_left=np.zeros(shape,bool)
    expected=np.zeros(shape)
    patterns={}
    depth_sum=0
    for t,tree in enumerate(trees):
        size=len(tree["left_children"])
        assert not any(tree["split_type"])
        left[t,:size],right[t,:size]=tree["left_children"],tree["right_children"]
        feature[t,:size],cut[t,:size],nan_left[t,:size]=tree["split_indices"],tree["split_conditions"],tree["default_left"]
        cover=np.asarray(tree["sum_hessian"],float)
        node_path=[()]*size
        depth=np.zeros(size,int)
        for n in range(size):
            l,r=left[t,n],right[t,n]
            if l<0:
                continue
            path=tuple(sorted(set((*node_path[n],int(feature[t,n])))))
            ix=patterns.setdefault(path,len(patterns))
            for nxt in (l,r):
                node_path[nxt]=path
                group[t,nxt]=ix
                depth[nxt]=depth[n]+1
        depth_sum+=int(depth.max())
        for n in reversed(range(size)):
            l,r=left[t,n],right[t,n]
            if l<0:
                expected[t,n]=cut[t,n]
            else:
                den=cover[l]+cover[r]
                if den<=0:
                    raise ValueError("Nonpositive child cover")
                expected[t,n]=(cover[l]*expected[t,l]+cover[r]*expected[t,r])/den
    p=row.shape[1]
    f=Forest(left,right,feature,cut,nan_left,expected,heads,group,list(patterns),depth_sum,p,c,np.zeros(c))
    native=model.get_booster().predict(DMatrix(row),output_margin=True,strict_shape=True)
    f.offset=native[0]-f.predict(row,np.arange(p))[0]
    return f


def as_scores(margins,task):
    if task=="regression":
        return margins[:,0]
    if margins.shape[1]==1:
        pos=expit(margins[:,0])
        return np.column_stack((1-pos,pos))
    return softmax(margins,axis=1)


class Game:
    def __init__(self,a,incidence):
        self.a=a
        self.incidence=incidence
        self.full=np.asarray(a.sum(axis=1)).ravel()
        self.norm=float(np.linalg.norm(self.full))
        self.p=incidence.shape[1]
        by_feature=incidence.tocsc()
        self.groups=[by_feature.indices[by_feature.indptr[j]:by_feature.indptr[j+1]]
                     for j in range(self.p)]

    def effects(self,active,j):
        ids=self.groups[j]
        ids=ids[active[ids]]
        return ids,np.asarray(self.a[:,ids].sum(axis=1)).ravel()

    def search(self,method,order,rho=.2):
        active=np.ones(self.a.shape[1],bool)
        keep=np.ones(self.p,bool)
        residual=np.zeros_like(self.full)
        limit=rho*self.norm+1e-12*max(1.,self.norm)
        n_checks=0
        rejected=0
        n_ballots=0
        comparisons=0
        if method=="gain_sweep":
            for _ in range(self.p):
                before=keep.sum()
                for j in order[::-1]:
                    if not keep[j]:
                        continue
                    ids,delta=self.effects(active,j)
                    n_checks+=1
                    candidate=residual+delta
                    if np.linalg.norm(candidate)<=limit:
                        keep[j]=False
                        active[ids]=False
                        residual=candidate
                if keep.sum()==before:
                    break
        elif method=="exact_greedy":
            delta=(self.a@self.incidence).toarray()
            while keep.any():
                # Candidate losses share the current residual. No new tree predictions.
                costs=np.einsum("ij,ij->j",delta,delta)+2*(residual@delta)+residual@residual
                costs[~keep]=np.inf
                j=int(np.lexsort((-np.argsort(order),costs))[0])
                n_checks+=int(keep.sum())
                if costs[j]>limit**2:
                    break
                ids=self.groups[j]
                ids=ids[active[ids]]
                residual+=delta[:,j]
                delta-=(self.a[:,ids]@self.incidence[ids]).toarray()
                keep[j]=False
                active[ids]=False
        elif method=="votes":
            weights=np.asarray(self.a.T@(self.full/max(self.norm,1e-100)))[None,:]
            projected=np.zeros(1)
            marginals=np.asarray(self.incidence.T@weights.T).T.copy()
            n_ballots=1
            checked=np.zeros(self.p,bool)
            while keep.any():
                bounds=np.abs(projected[:,None]+marginals).max(axis=0)
                comparisons+=n_ballots*int(keep.sum())
                bounds[~keep|checked]=np.inf
                j=int(np.lexsort((-np.argsort(order),bounds))[0])
                if bounds[j]>limit:
                    break
                ids,delta=self.effects(active,j)
                candidate=residual+delta
                n_checks+=1
                norm=np.linalg.norm(candidate)
                if norm<=limit:
                    projected+=marginals[:,j]
                    marginals-=np.asarray(self.incidence[ids].T@weights[:,ids].T).T
                    residual=candidate
                    keep[j]=False
                    active[ids]=False
                    checked[:]=False
                else:
                    checked[j]=True
                    direction=candidate/norm
                    w=np.asarray(self.a.T@direction).ravel()
                    weights=np.vstack((weights,w))
                    projected=np.r_[projected,direction@residual]
                    marginal=np.asarray(self.incidence.T@(w*active)).ravel()
                    if projected[-1]+marginal[j]<norm-1e-9*max(1.,norm):
                        raise ArithmeticError("Rejected residual failed to separate its support")
                    marginals=np.vstack((marginals,marginal))
                    n_ballots+=1
                    rejected+=1
        else:
            raise ValueError(method)
        assert np.linalg.norm(residual)<=limit+1e-10*max(1.,self.norm)
        return np.flatnonzero(keep),dict(checks=n_checks,cut_rejections=rejected,ballots=n_ballots,
            directional_comparisons=comparisons)
