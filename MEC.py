import numpy as np
from math import log

def get_f(X, paras):
    U = paras.U
    S = paras.S
    f_server = paras.f_server
    f_alloc = np.zeros((U, S))
    eta = paras.eta
    for s in range(S):
        sum_eta = 0
        for u in range(U):
            if X[u, s] == 1:
                sum_eta += eta[u] ** 0.5
        for u in range(U):
            if X[u, s] == 1:
                f_alloc[u, s] = f_server[s] * eta[u] ** 0.5 / sum_eta
    return f_alloc

def get_p(X, paras):
    P_max = paras.P_max
    U = paras.U
    S = paras.S
    zeta = paras.zeta
    lmd = paras.lmd
    miu = paras.miu
    W = paras.W
    P_alloc = []
    for u in range(U):
        s = -1
        for j in range(S):
            if X[u, j] == 1:
                s = j
                break

        if s == -1:
            P_alloc.append(0)
        else:
            a = 0.001 * P_max[u]
            b = P_max[u]
            p_tmp = (a + b) / 2
            value_p = (zeta[u] + lmd[u] * p_tmp) / (W * log(1 + miu[u, s] * p_tmp, 2))
            value_a = (zeta[u] + lmd[u] * a) / (W * log(1 + miu[u, s] * a, 2))
            value_b = (zeta[u] + lmd[u] * b) / (W * log(1 + miu[u, s] * b, 2))
            eps = 1

            while eps > 0.00005 * P_max[u]:
                if value_b > value_a:
                    b = p_tmp
                    value_b = value_p
                else:
                    a = p_tmp
                    value_a = value_p
                eps = abs(p_tmp - (a + b) / 2)
                p_tmp = (a + b) / 2
                value_p = (zeta[u] + lmd[u] * p_tmp) / (W * log(1 + miu[u, s] * p_tmp, 2))
            if abs(p_tmp - P_max[u]) <= P_max[u] * 0.00005:
                p_tmp = P_max[u]
            if abs(p_tmp - P_max[u] * 0.001) <= P_max[u] * 0.00005:
                p_tmp = 0
            P_alloc.append(p_tmp)

    return P_alloc

def get_J(X, paras):
    P_alloc = get_p(X, paras)
    f_alloc = get_f(X, paras)
    alpha_t = paras.alpha_t
    T_local = paras.T_local
    E_local = paras.E_local
    d = paras.d
    c = paras.c
    U = paras.U
    S = paras.S
    W = paras.W
    zigma = paras.zigma
    H = paras.H
    J = 0
    for u in range(U):
        s = -1
        for j in range(S):
            if X[u, j] == 1:
                s = j
                break
        if s != -1:
            T_exe = c[u] / f_alloc[u, s]
            R = W * log(1 + P_alloc[u] * H[u, s] / zigma, 2)
            T_up = d[u] / R
            T_edge = T_exe + T_up
            E_edge = P_alloc[u] * T_up
            J += alpha_t * (T_local[u] - T_edge) / T_local[u] + (1 - alpha_t) * (E_local[u] - E_edge) / E_local[u]
            # print(alpha_t * (T_local[u] - T_edge) / T_local[u])
            # print(T_local[u])
            # print(T_up)
            # print(T_exe)
            # print((1 - alpha_t) * (E_local[u] - E_edge) / E_local[u])
            # print('\n')
    return J

def get_J_pre(X, paras):
    P_alloc = get_p(X, paras)
    f_alloc = get_f(X, paras)
    alpha_t = paras.alpha_t
    T_local = paras.T_local
    E_local = paras.E_local
    d = paras.d
    c = paras.c
    U = paras.U
    S = paras.S
    W = paras.W
    zigma = paras.zigma
    H = paras.H
    J = np.zeros((U, S))
    for u in range(U):
        s = -1
        for j in range(S):
            if X[u, j] == 1:
                s = j
                break
        if s != -1:
            T_exe = c[u] / f_alloc[u, s]
            R = W * log(1 + P_alloc[u] * H[u, s] / zigma, 2)
            T_up = d[u] / R
            T_edge = T_exe + T_up
            E_edge = P_alloc[u] * T_up
            J[u, s] = alpha_t * (T_local[u] - T_edge) / T_local[u] + (1 - alpha_t) * (E_local[u] - E_edge) / E_local[u]
            # print(alpha_t * (T_local[u] - T_edge) / T_local[u])
            # print(T_local[u])
            # print(T_up)
            # print(T_exe)
            # print((1 - alpha_t) * (E_local[u] - E_edge) / E_local[u])
            # print('\n')
    return J
