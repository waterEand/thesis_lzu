import numpy as np
import scipy as sp
import cmath
from scipy.special import comb, perm
from signal_vectors import data_matrix
from scipy.sparse import csr_matrix
from numpy import linalg as LA

'''
data matrix X with shape of (n, n) is symmetric.
sketching matrix B : (2R, n_c)
'''

n = 10 # initialize n := min(n1, n2) 
R = 20 # number of measurements
n_c = int(comb(n, 2) + n) # n_culde

'''sketch!'''

def H_measure(row, col, k):
    res = np.zeros((row, col))
    for i in range(col):
        rvs = sp.stats.bernoulli(1).rvs
        S = sp.sparse.random(row, 1, density=k/row, data_rvs=rvs)
        res[:, i] = S.toarray().reshape((row, ))
    return res

def S_matrix(n, W):
    S = np.zeros((2, n), dtype=complex)
    for i in range(n):
        S[:, i] = np.array([[1], [W**(i)]], dtype=complex).reshape(2, )
    return S

# Kronecker product
def kron_product(a, b):
    res = np.empty(0)
    for elem in a:
        res = np.append(res, np.dot(elem, b.T))
    return res.T

# construct B
def B_matrix(H, S, n):
    B = np.empty((2 * H.shape[0], H.shape[1]), dtype=complex)
    for i in range(n):
        B[:, i] = kron_product(H[:, i], S[:, i])
    return B

def isSingleton(y, j, n):
    if (y[2*j].real == 0):
        return 0
    l_hat = cmath.phase(y[2*j+1]/y[2*j]) / (2*np.pi/n)
    if (abs(round(l_hat) - l_hat) < 0.001): 
        return l_hat
    return 0

'''
sketch done!
peeling decoder
'''

def peeling_decoder():
    X_org = data_matrix(n, n, r=1, k=3, isSymm=True)
    x_org = np.empty(0)  # vectorized upper-triangle part of X
    for i in range(n):
        for j in range(i, n):
            x_org = np.append(x_org, X_org[i][j])

    # print('The vectorized part is : ')
    # print(x_org)

    H = H_measure(R, n_c, 2)
    W = cmath.exp(2 * np.pi * cmath.sqrt(-1) / n_c)
    S = S_matrix(n_c, W)
    B = B_matrix(H, S, n_c)
    y = np.dot(B, x_org)

    x = np.zeros(n_c, dtype=complex) # estimate of x_org
    set_R = set(range(R)) # set of right nodes to be checked

    while (len(set_R) > 0):
        R_temp = set_R.copy() # R' <- R

        for j in set_R:

            if isSingleton(y, j, n_c) == 0: # not singleton
                R_temp.remove(j)

            else: # singleton

                l_hat = round(isSingleton(y, j, n_c))

                x[l_hat] = y[2*j]
                H[j, l_hat] = 0
                C = set()
                for i in range(R):
                    if (H[i, l_hat] == 1):
                        C.add(i)
                for c in C:
                    y[2*c] -= x[l_hat]
                    y[2*c+1] -= x[l_hat] * cmath.exp(2*cmath.pi*cmath.sqrt(-1)*(l_hat-1) / n_c)
                    if (not c in R_temp):
                        R_temp.add(c)
                R_temp.remove(j)

        set_R = R_temp.copy()
    x = x.real

    # print('\nEstimate of the vectorized signal x_hat: ')
    # print(x)

    err = LA.norm(x-x_org) / LA.norm(x_org)
    print("The error is : ", err)

if __name__ == '__main__':
    peeling_decoder()