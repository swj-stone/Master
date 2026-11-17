"""Exact marginal votes for a fitted additive predictor under squared loss."""
from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class AdditiveFit:
    imputer: SimpleImputer
    scaler: StandardScaler
    knots: np.ndarray
    basis_scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    center: np.ndarray

    def parts(self, x):
        z = self.scaler.transform(self.imputer.transform(x))
        basis = expand(z, self.knots) / self.basis_scale
        return np.einsum("npb,pbc->npc",basis,self.coefficients,optimize=True)

    def restrict(self, cols):
        cols=np.asarray(cols,dtype=int)
        omit=np.ones(len(self.center),dtype=bool)
        omit[cols]=False
        return SparseAdditive(self.imputer.statistics_[cols].copy(),self.scaler.mean_[cols].copy(),
                             self.scaler.scale_[cols].copy(),self.knots[:,cols].copy(),
                             self.coefficients[cols]/self.basis_scale[cols,:,None],
                             self.intercept+self.center[omit].sum(axis=0))


@dataclass
class SparseAdditive:
    fill: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    knots: np.ndarray
    slopes: np.ndarray
    intercept: np.ndarray

    def predict(self,x):
        if x.ndim!=2 or x.shape[1]!=len(self.fill):
            raise ValueError("Pass exactly the retained input columns in stored order")
        z=(np.where(np.isnan(x),self.fill,x)-self.mean)/self.scale
        return self.intercept + np.einsum("npb,pbc->nc",expand(z,self.knots),self.slopes,optimize=True)


def expand(x, knots):
    if len(knots) == 0:
        return x[:,:,None]
    bends = np.maximum(x[:,:,None] - knots.T[None,:,:], 0.)
    return np.concatenate((x[:,:,None], bends),axis=2)


def fit_additive(x, y, task, kind):
    imputer = SimpleImputer(strategy="median",keep_empty_features=True)
    scaler = StandardScaler()
    z = scaler.fit_transform(imputer.fit_transform(x))
    if task == "classification":
        y = np.eye(np.max(y)+1)[y]
    else:
        y = y.reshape(-1,1)
    knots = np.quantile(z,[.25,.5,.75],axis=0) if kind=="spline" else np.empty((0,z.shape[1]))
    basis = expand(z,knots)
    scale = basis.std(axis=0)
    scale[scale<1e-12]=1.
    design = (basis/scale).reshape(len(z),-1)
    model = Ridge(alpha=1.).fit(design,y)
    coefficients = np.asarray(model.coef_).reshape(y.shape[1],z.shape[1],-1).transpose(1,2,0)
    center=np.einsum("pb,pbc->pc",basis.mean(axis=0)/scale,coefficients)
    return AdditiveFit(imputer,scaler,knots,scale,coefficients,np.atleast_1d(model.intercept_),center)


def quadratic(parts, target):
    """G_ij=<a_i,a_j>, d_i=2<r,a_i>-G_ii. Inner products average over rows."""
    n,p,c = parts.shape
    matrix = parts.transpose(0,2,1).reshape(n*c,p)
    gram = (matrix.T @ matrix)/n
    return gram, 2*(matrix.T @ target.reshape(-1))/n - np.diag(gram)


def quadratic_order(gram, single, rule):
    """Backward elimination on the current active game; ties prefer deleting the last ID."""
    p = len(single)
    interactions = gram.copy()
    np.fill_diagonal(interactions,0.)
    if rule == "energy":
        return np.lexsort((np.arange(p),-np.diag(gram))), []
    if rule == "min":
        weights = 2*np.maximum(0,interactions)
    elif rule == "pooled":
        weights = interactions
    elif rule == "point":
        weights = 2*interactions
    else:
        raise ValueError(rule)
    score = single - weights.sum(axis=1)
    removed, votes = [], []
    for _ in range(p):
        j = p - 1 - int(np.argmin(score[::-1]))
        votes.append(float(score[j]))
        removed.append(j)
        score += weights[:,j]
        score[j] = np.inf
    return np.array(removed[::-1]),votes


def rank_additive(x, y, z, truth, task, kind, rule):
    model = fit_additive(x,y,task,kind)
    center = model.center
    parts = model.parts(z) - center
    baseline = model.intercept + center.sum(axis=0)
    target = np.eye(len(baseline))[truth] if task=="classification" else truth[:,None]
    gram,single = quadratic(parts,target-baseline)
    order,votes = quadratic_order(gram,single,rule)
    return order,dict(minimum_path=votes)
