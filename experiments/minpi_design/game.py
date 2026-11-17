"""Exact refitted-head regret under exchangeable input availability.

Whitening each penalized feature block makes the singleton matrix identity.
An arbitrary size uses a convex combination of the identity and the intact
matrix. All arrays are indexed by raw feature blocks, including after deletion.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def _expand(z, knots):
    if not len(knots):
        return z[:, :, None]
    return np.concatenate([z[:, :, None], np.maximum(z[:, :, None]-knots.T, 0.)], axis=2)


@dataclass
class Design:
    imputer: object
    scaler: object
    knots: np.ndarray
    center: np.ndarray
    whitening: np.ndarray

    @classmethod
    def fit(cls, x, kind):
        imputer = SimpleImputer(strategy="median",keep_empty_features=True)
        scaler = StandardScaler()
        z = scaler.fit_transform(imputer.fit_transform(x))
        knots = np.quantile(z,[1/3,2/3],axis=0) if kind=="spline" else np.empty((0,z.shape[1]))
        phi = _expand(z,knots)
        center = phi.mean(axis=0)
        phi -= center
        n, p, d = phi.shape
        cov = np.einsum("npd,npe->pde",phi,phi) / n
        chol = np.linalg.cholesky(cov + np.eye(d)[None,:,:]/n)
        whitening = np.linalg.inv(chol).transpose(0,2,1)
        return cls(imputer,scaler,knots,center,whitening), cls._whiten(phi,whitening)

    @staticmethod
    def _whiten(phi, whitening):
        return np.einsum("npj,pjk->npk",phi,whitening).reshape(len(phi),-1)

    def transform(self,x):
        z = self.scaler.transform(self.imputer.transform(x))
        return self._whiten(_expand(z,self.knots)-self.center,self.whitening)


@dataclass
class Profile:
    values: np.ndarray
    vectors: np.ndarray
    projected: np.ndarray
    mass: np.ndarray
    reference: np.ndarray

    def state(self, t):
        denominator = 1. + t * (self.values-1.)
        coef = self.vectors @ (self.projected/denominator[:,None])
        val = (1.-t)*self.reference[0] + t*self.reference[1] - np.sum(self.mass/denominator)
        return float(val), coef

    def slope(self,t):
        diff = self.values-1.
        return self.reference[1]-self.reference[0]+np.sum(self.mass*diff/(1.+t*diff)**2)

    def optimum(self):
        if self.slope(0.) <= 0.:
            return 0.
        if self.slope(1.) >= 0.:
            return 1.
        lo,hi = 0.,1.
        for _ in range(38):
            mid = (lo+hi)/2
            if self.slope(mid)>0:
                lo=mid
            else:
                hi=mid
        return (lo+hi)/2


class Game:
    def __init__(self, z, y, d):
        self.d = d
        self.p = z.shape[1]//d
        self.gram = z.T @ z / len(z)
        # The block diagonal of a whitened penalized fit Gram is exactly I.
        for j in range(self.p):
            b = slice(j*d,(j+1)*d)
            self.gram[b,b] = np.eye(d)
        self.cross = z.T @ y / len(z)
        self.reference = np.array([np.sum(self.cross**2),
            np.sum(self.cross*cho_solve(cho_factor(self.gram),self.cross))])

    def coordinates(self, raw):
        return (np.asarray(raw,dtype=int)[:,None]*self.d+np.arange(self.d)).ravel()

    def profile(self, keep):
        ix = self.coordinates(keep)
        vals, vec = eigh(self.gram[np.ix_(ix,ix)],check_finite=False,driver="evd")
        project = vec.T @ self.cross[ix]
        return Profile(vals,vec,project,np.einsum("ic,ic->i",project,project),self.reference)

    def fixed_fit(self, keep, t):
        ix = self.coordinates(keep)
        h = t*self.gram[np.ix_(ix,ix)]
        h.flat[::len(h)+1] += 1.-t
        g = self.cross[ix]
        coef = cho_solve(cho_factor(h,check_finite=False),g,check_finite=False)
        return coef

    def head(self, keep, robust):
        if not robust:
            return self.fixed_fit(keep,1.)
        pr = self.profile(keep)
        return pr.state(pr.optimum())[1]

    def deletion_path(self, method):
        """Trace nested supports. Exact robust rescoring uses a dual lower bound."""
        keep = np.arange(self.p)
        path = [keep.copy()]
        exact_candidates = possible = 0
        while len(keep)>1:
            pr = self.profile(keep)
            t = pr.optimum() if method in ("min","min_bound","min_exhaustive") else .5 if method=="pooled" else 1.
            obj,beta = pr.state(t)
            inv = (pr.vectors / (1.+t*(pr.values-1.))) @ pr.vectors.T
            shape = (len(keep),self.d,len(keep),self.d)
            diag = inv.reshape(shape)[np.arange(len(keep)),:,np.arange(len(keep)),:]
            betas = beta.reshape(len(keep),self.d,-1)
            sol = np.linalg.solve(diag,betas)
            lower = obj+np.einsum("pdc,pdc->p",sol,betas)
            possible += len(keep)
            j = int(np.argmin(lower))
            if method in ("min","min_exhaustive"):
                best = np.inf
                tol = 3e-10*max(1.,np.max(self.reference))
                for candidate in np.argsort(lower,kind="stable"):
                    if method=="min" and lower[candidate]>best+tol:
                        break
                    exact_candidates+=1
                    new = self.profile(np.delete(keep,candidate))
                    value = new.state(new.optimum())[0]
                    if value<best-tol or (abs(value-best)<=tol and keep[candidate]<keep[j]):
                        best,j=value,int(candidate)
            keep=np.delete(keep,j)
            path.append(keep.copy())
        return path, dict(exact=exact_candidates, possible=possible)


def endpoint_losses(z, y, cols, beta, d):
    """Two normalized endpoint risks, omitting a shared constant ||y||²."""
    blocks = z.reshape(len(z),-1,d)[:,cols,:]
    parts = np.einsum("npd,pdc->npc",blocks,beta.reshape(len(cols),d,-1))
    pred = parts.sum(axis=1)
    linear = 2*np.sum(y*pred)
    return pred, np.array([np.einsum("npc,npc->",parts,parts)-linear,
                           np.einsum("nc,nc->",pred,pred)-linear])/len(y)
