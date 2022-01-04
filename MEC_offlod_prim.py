import numpy as np
from MEC import Bisection
from MEC import J

def remove(X, u, s, j):
    X_tmp = X.copy()
    X_tmp[u, s, j] = 0
    return X_tmp

def exchange(X, u, s, j):
    X_tmp = X.copy()
    X_tmp[u, :, :] = 0
    X_tmp[:, s, j] = 0
    X_tmp[u, s, j] = 1
    return X_tmp

def offload(paras):
    mec = paras.mec
    U = paras.U
    S = mec.S
    N = mec.N
    X = np.zeros((U, S, N))
    umax = 0
    smax = 0
    jmax = 0
    J_pre = 0
    p = S*N*100
    for u in range(U):
        for s in range(S):
            for j in range(N):
                X_temp = X.copy()
                X_temp[u, s, j] += 1
                paras.update_theta(X_temp)
                P = Bisection(X_temp, paras)
                J_tmp = J(X_temp, P, paras)
                if J_tmp > J_pre:
                    umax = u
                    smax = s
                    jmax = j
                    J_pre = J_tmp
    X[umax, smax, jmax] = 1

    flag = True
    J_x = J_pre
    while flag:
        flag = False
        for u in range(U):
            for s in range(S):
                for j in range(N):
                    if X[u, s, j] == 1:
                        X_temp = remove(X, u, s, j)
                        paras.update_theta(X_temp)
                        P_tmp = Bisection(X_temp, paras)
                        J_tmp = J(X_temp, P_tmp, paras)
                        if J_tmp > (1 + 1/p)*J_x:
                            X = X_temp.copy()
                            J_x = J_tmp
                            flag = True
                    else:
                        X_temp = exchange(X, u, s, j)
                        paras.update_theta(X_temp)
                        P_tmp = Bisection(X_temp, paras)
                        J_tmp = J(X_temp, P_tmp, paras)
                        if J_tmp > (1 + 1/p)*J_x:
                            X = X_temp.copy()
                            J_x = J_tmp
                            flag = True
    return J_x, X