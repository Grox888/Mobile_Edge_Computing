import numpy as np
from math import log
from math import sqrt
from MEC_paras import Paras
from MEC_H_generate import genGain
from matplotlib import pyplot as plt

#to calculate the gama func
def Gama(X, P, paras):
    user_list = paras.u_list
    mec = paras.mec
    h_usj = paras.H
    zigma = paras.zigma
    U = paras.U
    S = mec.S
    N = mec.N
    gama = 0
    I_usj = np.zeros((U, S, N))
    theta = np.zeros((U, S))
    phi = []
    psai = []

    for s in range(S):
        for u in range(U):
            if sum(X[u, s, :]) == 1:
                for j in range(N):
                    for w in range(S):
                        if w == s:
                            continue
                        for k in range(U):
                            I_usj[u, s, j] += X[k, w, j]*user_list[k].Pu*h_usj[k, w, j]
    for u in range(U):
        for s in range(S):
            for j in range(N):
                theta[u, s] += h_usj[u, s, j]/(I_usj[u, s, j] + zigma**2)
    for u in range(U):
        phi.append(user_list[u].lamda
                   *user_list[u].beta_t
                   *user_list[u].du
                   *user_list[u].ful
                   /(user_list[u].cu*mec.W))
    phi = np.array(phi)
    for u in range(U):
        psai.append(user_list[u].lamda
                    *(1 - user_list[u].beta_t)
                    *user_list[u].du
                    /(user_list[u].K
                      *(user_list[u].ful**2)
                      *user_list[u].cu
                      *mec.W))
    psai = np.array(psai)
    for s in range(S):
        for u in range(U):
            if sum(X[u, s, :]) == 1:
                gama += (phi[u] + psai[u]*P[u])/log(1 + theta[u, s]*P[u], 2)
    return gama

def Gama_der(X, P, _u, paras):
    user_list = paras.u_list
    mec = paras.mec
    h_usj = paras.H
    zigma = paras.zigma
    U = paras.U
    S = mec.S
    N = mec.N
    I_usj = np.zeros((U, S, N))
    theta = np.zeros((U, S))
    phi = 0
    psai = 0
    for s in range(S):
        for u in range(U):
            if sum(X[u, s, :]) == 1:
                for j in range(N):
                    for w in range(S):
                        if w == s:
                            continue
                        for k in range(U):
                            I_usj[u, s, j] += X[k, w, j]*user_list[k].Pu*h_usj[k, w, j]
    for u in range(U):
        for s in range(S):
            for j in range(N):
                theta[u, s] += h_usj[u, s, j]/(I_usj[u, s, j] + zigma**2)

    phi = (user_list[_u].lamda
                *user_list[_u].beta_t
                *user_list[_u].du
                *user_list[_u].ful
                /(user_list[_u].cu*mec.W))

    psai = (user_list[_u].lamda
                *(1 - user_list[_u].beta_t)
                *user_list[_u].du
                /(user_list[_u].K
                  *(user_list[_u].ful**2)
                  *user_list[_u].cu
                  *mec.W))
    gama_der = 0
    for s in range(S):
        if sum(X[_u, s, :]) == 1:
            gama_der = psai*log(1 + theta[_u, s]*P, 2) - theta[_u, s]*(phi + psai*P)/((1 + theta[_u, s]*P)*log(2))
    return gama_der

#to calculate the Lambda func
def Lmd(X, paras):
    user_list = paras.u_list
    mec = paras.mec
    lmd = 0
    U = paras.U
    S = mec.S
    for s in range(S):
        eta_sqrt_sum = 0
        for u in range(U):
            if sum(X[u, s, :]) == 1:
                eta_sqrt_sum += sqrt(user_list[u].lamda*
                                     user_list[u].beta_t*
                                     user_list[u].ful)
        lmd += (eta_sqrt_sum**2)/mec.fs[s]
    return lmd

#to calculate the evaluating func J
def J(X, P, paras):
    user_list = paras.u_list
    mec = paras.mec
    U = paras.U
    S = mec.S
    gama = Gama(X, P, paras)
    lmd = Lmd(X, paras)
    J_value = - gama - lmd
    for s in range(S):
        for u in range(U):
            if sum(X[u, s, :]) == 1:
                J_value += user_list[u].lamda
    return J_value

def Bisection(X, paras):
    user_list = paras.u_list
    eps = paras.eps
    U = paras.U
    p_star = np.zeros(U)
    for u in range(U):
        if sum(X[u, :, :].ravel()) == 0:
            continue
        p_tmp = user_list[u].Pu
        der = Gama_der(X, p_tmp, u, paras)
        if der > 0:
            p1 = 0
            p2 = p_tmp
            while (p2 - p1) > eps:
                p_tmp = (p1 + p2)/2
                der = Gama_der(X, p_tmp, u, paras)
                if der <= 0:
                    p1 = p_tmp
                else:
                    p2 = p_tmp
            p_tmp = (p1 + p2)/2
        p_star[u] = p_tmp
    return p_star

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
    p = 100
    for u in range(U):
        for s in range(S):
            for j in range(N):
                X_temp = X.copy()
                X_temp[u, s, j] += 1
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
                        P_tmp = Bisection(X_temp, paras)
                        J_tmp = J(X_temp, P_tmp, paras)
                        if J_tmp > (1 + 1/p)*J_x:
                            X = X_temp.copy()
                            print(J_tmp)
                            J_x = J_tmp
                            flag = True
        for u in range(U):
            for s in range(S):
                for j in range(N):
                    if X[u, s, j] == 0:
                        X_temp = exchange(X, u, s, j)
                        P_tmp = Bisection(X_temp, paras)
                        J_tmp = J(X_temp, P_tmp, paras)
                        if J_tmp > (1 + 1/p)*J_x:
                            X = X_temp.copy()
                            print(J_tmp)
                            J_x = J_tmp
                            flag = True
    return J_x, X


#below are the testing codes:
loop = 10
res_J = []
res_U = []
for idx in range(loop):
    # below are the object structures
    # user(du, cu, Pu, lamda, beta_t, ful, K)
    # MEC(S, N, B, fs)
    # H, zigma, eps

    #Topology of physics model
    gapofserver = 50

    #parameters of MEC
    U = 5 + idx
    S = 4
    N = 3
    B = 20e7

    #parameters of users
    fs = [20e9 for i in range(S)]
    du = [420 * 1024 * 8 for i in range(U)]
    cu = [1e9 for i in range(U)]
    Pu = [0.1 for i in range(U)]
    lamda = [1 for i in range(U)]
    beta_t = [0.5 for i in range(U)]
    ful = [1e9 for i in range(U)]
    K = 1e-27

    #parameters of uplink
    H = genGain(U, S, N, gapofserver)
    zigma = 1e-6

    #parameter of error control
    eps = 0.0005

    # structure of Paras:
    # paras = {
    #     u_list,
    #     U,
    #     mec = {S, N, W, fs},
    #     H,
    #     zigma,
    #     eps
    # }

    paras = Paras(du, cu, Pu, lamda, beta_t, ful, K, U, S, N, B, fs, H, zigma, eps)
    J_X, X = offload(paras)
    res_J.append(J_X)
    res_U.append(U)

res_J = np.array(res_J)
res_U = np.array(res_U)

fig = plt.figure()
plt.plot(res_U, res_J, color='r', marker='o')
plt.show()
