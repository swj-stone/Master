"""Cover-weighted truncation of a fitted scikit-learn random forest on numeric features."""
from dataclasses import dataclass
import numpy as np
from numba import njit
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.impute import SimpleImputer
from experiments.minpi_rework.quota_compress_tree import Game


@njit(cache=True)
def _predict(x,lookup,left,right,feature,threshold,value):
    n,c=len(x),value.shape[2]
    result=np.zeros((n,c))
    for i in range(n):
        for t in range(len(left)):
            node=0
            while left[t,node]>=0:
                col=lookup[feature[t,node]]
                if col<0:
                    break
                node=left[t,node] if x[i,col]<=threshold[t,node] else right[t,node]
            for h in range(c):
                result[i,h]+=value[t,node,h]
    return result


@njit(cache=True)
def _groups(x,left,right,feature,threshold,value,group,depth_sum):
    c=value.shape[2]
    cap=len(x)*depth_sum*c
    rr=np.empty(cap,np.int32)
    cc=np.empty(cap,np.int32)
    vv=np.empty(cap,np.float64)
    pos=0
    for i in range(len(x)):
        for t in range(len(left)):
            node=0
            while left[t,node]>=0:
                child=left[t,node] if x[i,feature[t,node]]<=threshold[t,node] else right[t,node]
                for h in range(c):
                    diff=value[t,child,h]-value[t,node,h]
                    if diff!=0.:
                        rr[pos]=i*c+h
                        cc[pos]=group[t,child]
                        vv[pos]=diff
                        pos+=1
                node=child
    return rr[:pos],cc[:pos],vv[:pos]


@dataclass
class PanelForest:
    model: object
    medians: np.ndarray
    classification: bool
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    group: np.ndarray
    sets: list
    depth_sum: int

    @property
    def empty(self):
        return self.value[:,0].sum(axis=0)

    @property
    def p(self):
        return len(self.medians)

    def _array(self,x,cols):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=len(cols):
            raise ValueError("Pass a matrix containing only the supplied input columns")
        return np.ascontiguousarray(np.where(np.isnan(x),self.medians[cols],x),dtype=np.float32)

    def predict(self,x,cols=None):
        cols=np.arange(self.p) if cols is None else np.asarray(cols,dtype=int)
        if np.any((cols<0)|(cols>=self.p)) or len(np.unique(cols))!=len(cols):
            raise ValueError("Columns must be distinct valid original indices")
        lookup=np.full(self.p,-1,int)
        lookup[cols]=np.arange(len(cols))
        raw=_predict(self._array(x,cols),lookup,self.left,self.right,self.feature,self.threshold,self.value)
        return raw if self.classification else raw[:,0]

    def native(self,x):
        fixed=self._array(x,np.arange(self.p))
        return self.model.predict_proba(fixed) if self.classification else self.model.predict(fixed)

    def game(self,x):
        fixed=self._array(x,np.arange(self.p))
        c=self.value.shape[2]
        blocks=[]
        # Chunk temporary coordinate storage, particularly for multiclass forests.
        block=max(1,min(512,1000000//max(1,self.depth_sum*c)))
        for start in range(0,len(x),block):
            chunk=fixed[start:start+block]
            i,j,v=_groups(chunk,self.left,self.right,self.feature,self.threshold,self.value,self.group,self.depth_sum)
            matrix=sparse.coo_matrix((v,(i,j)),shape=(len(chunk)*c,len(self.sets))).tocsr()
            matrix.eliminate_zeros()
            blocks.append(matrix)
        a=sparse.vstack(blocks,format="csc") if blocks else sparse.csc_matrix((0,len(self.sets)))
        rows=[i for i,s in enumerate(self.sets) for _ in s]
        cols=[j for s in self.sets for j in s]
        inc=sparse.csr_matrix((np.ones(len(cols)),(rows,cols)),shape=(len(self.sets),self.p))
        return Game(a,inc)


def fit(x,y,task,seed,n_estimators=100,max_depth=4,sample_weight=None):
    """Fit an imputer and numeric RF on these rows; E+game.full reconstructs its predictions."""
    if task not in ("classification","regression","binary","multiclass"):
        raise ValueError(f"Unsupported task {task}")
    classification=task!="regression"
    imputer=SimpleImputer(strategy="median",keep_empty_features=True)
    train=np.asarray(imputer.fit_transform(x),dtype=np.float32)
    factory=RandomForestClassifier if classification else RandomForestRegressor
    model=factory(n_estimators=n_estimators,max_depth=max_depth,n_jobs=1,random_state=seed)
    model.fit(train,y,sample_weight=sample_weight)
    trees=[learner.tree_ for learner in model.estimators_]
    shape=(len(trees),max(t.node_count for t in trees))
    c=len(model.classes_) if classification else 1
    left=np.full(shape,-1,np.int32)
    right=left.copy()
    feature=left.copy()
    group=left.copy()
    threshold=np.zeros(shape)
    value=np.zeros((*shape,c))
    patterns={}
    for t,est in enumerate(trees):
        z=est.node_count
        left[t,:z],right[t,:z],feature[t,:z],threshold[t,:z]=est.children_left,est.children_right,est.feature,est.threshold
        local=est.value[:,0,:].copy()
        if classification:
            local/=local.sum(axis=1,keepdims=True)
        value[t,:z]=local/len(trees)
        paths=[()]*z
        for node in range(z):
            l,r=left[t,node],right[t,node]
            if l<0:
                continue
            path=tuple(sorted(set((*paths[node],int(feature[t,node])))))
            ix=patterns.setdefault(path,len(patterns))
            paths[l]=paths[r]=path
            group[t,l]=group[t,r]=ix
        internal=est.children_left>=0
        if internal.any():
            l,r=est.children_left[internal],est.children_right[internal]
            wl,wr=est.weighted_n_node_samples[l,None],est.weighted_n_node_samples[r,None]
            if not np.allclose(local[internal],(wl*local[l]+wr*local[r])/(wl+wr),rtol=1e-8,atol=1e-10):
                raise ArithmeticError("Stored RF node disagrees with training-weighted descendant predictions")
    return PanelForest(model,imputer.statistics_.copy(),classification,left,right,feature,threshold,value,group,
        list(patterns),sum(t.max_depth for t in trees))
