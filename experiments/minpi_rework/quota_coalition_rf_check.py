"""Brute-force all supports, including zero-width inputs, against native sklearn trees."""
import numpy as np
from sklearn.impute import SimpleImputer
from experiments.minpi_rework.quota_coalition_rf import fit


def reference(f,x,cols):
    lookup={j:i for i,j in enumerate(cols)}
    vals=[]
    for row in x:
        contributions=[]
        for est in f.model.estimators_:
            t=est.tree_
            node=0
            while t.children_left[node]>=0 and t.feature[node] in lookup:
                j=t.feature[node]
                val=row[lookup[j]]
                if np.isnan(val):
                    val=f.medians[j]
                node=t.children_left[node] if np.float32(val)<=t.threshold[node] else t.children_right[node]
            def leaves(i):
                if t.children_left[i]<0:
                    pred=t.value[i,0].copy()
                    if f.classification:
                        pred/=pred.sum()
                    return pred*t.weighted_n_node_samples[i]
                return leaves(t.children_left[i])+leaves(t.children_right[i])
            contributions.append(leaves(node)/t.weighted_n_node_samples[node])
        vals.append(np.mean(contributions,axis=0))
    a=np.asarray(vals)
    return a if f.classification else a[:,0]


def check():
    rng=np.random.default_rng(1063)
    x=rng.normal(size=(155,5))
    xf=rng.normal(loc=1.3,size=(17,5))
    y=2*x[:,0]+x[:,1]*x[:,2]+rng.normal(size=len(x))
    w=np.linspace(.1,4.,len(x))
    x[::3,1]=np.nan
    x[:,4]=np.nan
    xf[::2,2]=np.nan
    xf[1,0]=np.nan
    for task,target in (("regression",y),("classification",y>0),("classification",np.digitize(y,[-1,1]))):
        f=fit(x,target,task,41,n_estimators=5,max_depth=3,sample_weight=w)
        native_x=SimpleImputer(strategy="median",keep_empty_features=True).fit(x).transform(xf)
        original=f.model.predict_proba(native_x) if f.classification else f.model.predict(native_x)
        np.testing.assert_allclose(f.predict(xf),original,atol=1e-12)
        np.testing.assert_allclose(f.native(xf),original)
        g=f.game(xf)
        allcols=np.arange(5)
        for bits in range(32):
            cols=allcols[((bits>>allcols)&1).astype(bool)][::-1]
            raw=f.predict(xf[:,cols],cols)
            np.testing.assert_allclose(raw,reference(f,xf[:,cols],cols),atol=1e-12)
            missing=np.setdiff1d(allcols,cols)
            active=np.asarray(g.incidence[:,missing].sum(axis=1)).ravel()==0
            d=np.asarray(g.a[:,active].sum(axis=1)).reshape(len(xf),-1)+f.empty
            np.testing.assert_allclose(raw,d if f.classification else d[:,0],atol=1e-12)
            if f.classification:
                np.testing.assert_allclose(raw.sum(axis=1),1.,atol=1e-12)
    constant=fit(x,np.ones(len(x)),"regression",99,n_estimators=3)
    g=constant.game(xf)
    assert g.a.shape==(len(xf),0) and np.all(g.full==0.)
    np.testing.assert_allclose(constant.predict(xf[:,:0],[]),1.)
    print("RF native, weighted descendants, and exact path cache agree: 96 exhaustive supports and a constant forest.")


if __name__=="__main__":
    check()
