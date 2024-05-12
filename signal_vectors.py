import numpy as np
import scipy as sp

### generate sparse matrix X_0  (X = X_0 + W)

'''
shape of data matrix X : (n1, n2)
r: the rank of X
k: signal vectors are k-sparse
'''

def signal_vec(n, k):
    rvs = sp.stats.norm(loc=0, scale=1).rvs
    S = sp.sparse.random(n, 1, density=k/n, data_rvs=rvs)
    return S.toarray()

def outliers(n, k, var=100):
    rvs = sp.stats.norm(loc=0, scale=var).rvs
    S = sp.sparse.random(n, 1, density=k / n, data_rvs=rvs)
    return S.toarray()

def data_matrix(n1, n2, r, k, isSymm):
    X = np.zeros((n1, n2))

    for i in range(r):
        u = signal_vec(n1, k)
        v = signal_vec(n2, k)
        if (isSymm):
            X += np.dot(u, u.T)
        else:
            X += np.dot(u, v.T)

    return X
