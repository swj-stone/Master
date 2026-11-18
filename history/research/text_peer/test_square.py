from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from research.text_peer.square import square_scores


def test_every_expected_coalition_gain_matches_affine_endpoints_with_absence():
    X=np.array([[0,1,0,1],[1,1,0,0],[0,1,1,1],[0,0,0,1],[0,0,1,0],[1,0,1,1],[1,1,0,0],[1,0,0,0]])
    y=np.array([0,0,0,0,0,1,1,1])
    gains,slope=square_scores(csr_matrix(X),y)
    features=X.shape[1]; all_cols=set(range(features))
    margins=[]
    for i,row in enumerate(X):
        held=np.arange(len(y))!=i
        own=(y==y[i]) & held; other=(y!=y[i])
        weight=np.mean(y[held]==y[i])
        direct=[((s:=np.sum(X[own,j]==row[j])/weight) - (d:=np.sum(X[other,j]==row[j])/(1-weight))) / (s+d+2) for j in range(features)]
        margins.append(direct)
    margins=np.array(margins)
    probability=np.array([0.5/np.sum(y==label) for label in y])
    z=margins.sum(axis=1);alpha=max(0.,np.dot(probability,z)/np.dot(probability,z*z))
    np.testing.assert_allclose(slope,alpha)
    def value(S):return -np.dot(probability, (1-alpha*margins[:,list(S)].sum(axis=1))**2)
    for j in all_cols:
        for k in range(features):
            permutations=list(combinations(sorted(all_cols-{j}),k))
            gain=np.mean([value((*S,j))-value(S) for S in permutations])
            predicted=gains[0,j] + (gains[1,j]-gains[0,j]) * k/(features-1)
            np.testing.assert_allclose(gain,predicted,atol=3e-15)
