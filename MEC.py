import numpy as np
from math import log
from math import sqrt



#to calculate the gama func
def Gama(X, P, paras):
    mec = paras.mec
    U = paras.U
    S = mec.S
    gama = 0
    theta = paras.theta
    phi = paras.phi
    psai = paras.psai
    for s in range(S):
        for u in range(U):
            if sum(X[u, s, :]) == 1:
                gama += (phi[u] + psai[u]*P[u])/log(1 + theta[u, s]*P[u], 2)
    return gama

def Gama_der(X, P, _u, paras):
    mec = paras.mec
    S = mec.S
    phi = paras.phi[_u]
    psai = paras.psai[_u]
    theta = paras.theta
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