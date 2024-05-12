import numpy as np
import time
from signal_vectors import signal_vec, outliers

def A_Gauss(m, n, mean, var):
    Matrix = np.random.normal(mean, var**0.5, size=[m, n])
    Matrix = np.mat(Matrix)
    return Matrix
def SoftThreshold(b, lambd):
    xx = np.maximum(np.abs(b) - lambd, 0)
    return np.multiply(np.sign(b),xx)

def PLAD(lambd, iterN, m, n, A, x0, e):
    b = A * x0 + e
    x = np.zeros((n, 1), dtype=complex)
    z = np.zeros((m, 1), dtype=complex)
    AA = np.dot(A.T, A)
    a, v = np.linalg.eig(AA)
    L = np.max(a)
    t = 1 / L ** 2
    for k in range(iterN):
        alfa = 1/L
        x = SoftThreshold(x - alfa * A.T * z, alfa * lambd)
        z = z + t * (A * x -b)
        for i in range(m):
            if z[i,0] > t:
                z[i, 0] = t
            if z[i,0] < -t:
                z[i, 0] = -t
    return x, x0, np.linalg.norm((x-x0), ord=2) / np.linalg.norm(x0, ord=2), max(np.abs(A.T*e))


n = 200
m = 100
lam = 0.006
k = 2
thr_err = 1e-2
num_epoch = 500

if __name__ == '__main__':
    num_succ_PLAD = 0
    time_sum = 0.

    for epoch in range(num_epoch):
        A = A_Gauss(m, n, 0, 1 / m)
        x0 = signal_vec(n, k)
        e = outliers(m,  3, 100)

        start_time = time.time()

        _, _, E_PLAD, _ = PLAD(lam, 2000, m, n, A, x0, e)

        end_time = time.time()
        time_sum += (end_time - start_time)

        # print(E_PLAD)
        if E_PLAD < thr_err:
            num_succ_PLAD += 1

        if (epoch + 1) % 100 == 0:
            print("time of succ in {} epochs: ".format(epoch+1), num_succ_PLAD)

    print("The average time cost of {} epoch is : ".format(num_epoch), time_sum/num_epoch)

