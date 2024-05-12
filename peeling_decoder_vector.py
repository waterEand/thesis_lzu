import random
from statistics import median
import time
import numpy as np
import scipy as sp
import cmath
from signal_vectors import signal_vec, outliers
from scipy.sparse import csr_matrix
from numpy import linalg as LA

n = 200 # 16
R = 100 # 9
k = 2 # sparsity of x
density_H = 3
num_outliers = 1
med_times = 2 # values peeled off
percent = 50
var_outliers = 100 # variance of outliers
num_epoch = 500
total_err = 0.

def input_vector(n, k):
    x = signal_vec(n, k)
    return np.squeeze(x)

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

def kron_product(a, b):
    res = np.empty(0)
    for elem in a:
        res = np.append(res, np.dot(elem, b.T))
    return res.T

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

def add_outliers(y_hat, y, num_outliers, med_times, var, R, W):
    for i in range(num_outliers):
        rand_row = random.randint(0, R-1)
        out_val = np.random.normal(0, var_outliers)
        y[rand_row] += out_val
        y_hat[rand_row*2] += out_val
        y_hat[rand_row*2+1] += out_val * W
    y_supp = np.transpose(np.nonzero(y_hat))
    y_supp_elem = []
    for index in y_supp:
        if index%2 == 0:
            y_supp_elem.append(abs(y_hat[index][0]))
    y_supp_elem = np.array(y_supp_elem)
    med = np.percentile(y_supp_elem, percent)
    for index in y_supp:
        if index%2 == 0:
            if abs(y_hat[index]) > med_times*med:
                y_hat[index] = 0
                y_hat[index+1] = 0.
    return y_hat

def peeling_decoder():
    total_err = 0.
    num_success = 0
    start_time = time.time()

    for epoch in range(num_epoch):
        x = input_vector(n, k)
        W = cmath.exp(2 * np.pi * cmath.sqrt(-1) / n)
        H = H_measure(R, n, density_H) # (R, n)
        S = S_matrix(n, W)
        B = B_matrix(H, S, n)
        y = np.dot(H, x)
        y = np.squeeze(y)
        y_hat = np.dot(B, x) # y_hat = np.zeros(R*2, dtype=complex)
        y_hat = np.squeeze(y_hat)
        y_hat = add_outliers(y_hat, y, num_outliers, med_times, var_outliers, R=R, W=W)
        x_hat = np.zeros(n)

        err = 1.
        while (err > 0.01):
            flag = False
            for r in range(R):
                if isSingleton(y_hat, r, n) != 0:
                    flag = True
                    l_hat = round(isSingleton(y_hat, r, n))
                    x_hat[l_hat] = y_hat[2*r].real
                    y_hat[2*r] = 0
                    y_hat[2*r+1] = 0
                    H[r, l_hat] = 0
                    for j in range(R):
                        if (H[j, l_hat] > 0 and j != r):
                            r1 = j
                            H[j, l_hat] = 0
                            y_hat[r1*2] = y_hat[r1*2] - x_hat[l_hat]
                            y_hat[r1*2+1] = y_hat[r1*2+1] - x_hat[l_hat] * np.power(W, l_hat)
                            break
                else:
                    continue
            if (np.all(y_hat.real < 1e-7)):
                break
            elif(not flag):
                print("There is no single-ton but still multi-ton in y_hat!\n")
                break

        err = LA.norm(x_hat - x) / LA.norm(x)
        total_err += err
        if err < 1e-3:
            num_success += 1
        if (epoch+1) % 100 == 0:
            print("The average error of {} epoch is : ".format(epoch+1), total_err/(epoch+1))
            print("time of succ: ", num_success)

    end_time = time.time()
    print("The average time cost of {} epoch is : ".format(num_epoch), (end_time - start_time)/num_epoch)

if __name__ == '__main__':
    peeling_decoder()