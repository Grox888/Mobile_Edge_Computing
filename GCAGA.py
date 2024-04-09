import random
import numpy as np
from MEC import get_J
from math import ceil

def init_ger(Ger_max, paras):
    U = paras.U
    S = paras.S
    N = paras.N
    dna = []
    for idv in range(Ger_max - 1):
        dna_idv = np.zeros((U, S), dtype=bool)
        for u in range(U):
            s = random.randint(0, S - 1)
            dna_idv[u, s] = 1
        dna.append(dna_idv)
    dna.append(np.zeros((U, S), dtype=bool))
    return np.array(dna)

def survival_of_fit(dna, Fittness_in, Ger_max):
    dna_new = []
    Fittness = Fittness_in.copy()
    mean = np.mean(Fittness)
    Fittness = Fittness - mean
    max = np.max(Fittness)
    min = np.min(Fittness)
    if max - min == 0:
        Fittness[:] = 1
    else:
        Fittness = (Fittness - min) / (max - min)
    Fittness_sum = np.sum(Fittness)
    Fittness_acc = np.zeros_like(Fittness)
    Fittness_sort = np.argsort(Fittness)

    L = Fittness.shape[0]
    min_1 = Fittness[Fittness_sort[0]]
    mean_1 = np.mean(Fittness)
    chosen = Fittness[Fittness_sort[L // 2]]

    # 计算自适应改变量
    delta = 0
    if min_1 - mean_1 != 0:
        delta = (mean_1 - chosen) / (2 * (mean_1 - min_1))

    for i in range(L):
        Fittness[Fittness_sort[i]] = Fittness[Fittness_sort[i]] * (0.7 + 0.6 * (i + 1) / L)
    Fittness_tmp = 0
    for i in range(L):
        Fittness_tmp += Fittness[Fittness_sort[i]]
        Fittness_acc[Fittness_sort[i]] = Fittness_tmp
    Fittness_acc /= Fittness_sum
    keep = []
    i = L - 1
    best = -1
    while len(keep) < Ger_max:
        b = Fittness_acc[Fittness_sort[i]]
        if i == 0:
            a = 0
        else:
            a = Fittness_acc[Fittness_sort[i - 1]]
        p = random.random()
        if p >= a and p < b:
            keep.append(Fittness_sort[i])
            if i > best:
                best = i
        i = (i + L - 1) % L
    best_final = 0
    for idv in range(Ger_max):
        dna_new.append(dna[keep[idv]])
        if keep[idv] == Fittness_sort[best]:
            best_final = idv
    return dna_new, best_final, delta

def get_server_bound(paras):
    U = paras.U
    S = paras.S
    N = paras.N

    X = np.zeros((U, S), dtype=bool)
    X_tmp = np.zeros((U, S), dtype=bool)
    J_all = np.zeros((U, S))
    for u in range(U):
        for s in range(S):
            X_tmp[u, s] = 1
            J_all_tmp = get_J(X_tmp, paras)
            if J_all_tmp > 0:
                J_all[u, s] = J_all_tmp
                X[u, s] = 1
            X_tmp[u, s] = 0

    Y = np.zeros(S)
    B = np.zeros(S, dtype=int)
    J = []
    for s in range(S):
        X_tmp[:, s] = X[:, s]
        Y[s] = get_J(X_tmp, paras)
        B[s] = np.sum(X_tmp[:, s])
        Ji = []
        J_tmp = J_all[:, s]
        J_tmp_sort = np.argsort(J_tmp)
        X_tmp[:, s] = False
        for u in range(U):
            if J_tmp[J_tmp_sort[u]] != 0:
                Ji.append(J_tmp[J_tmp_sort[u]])
        J.append(Ji)

    y = []
    for s in range(S):
        ys = []
        for i in range(B[s]):
            yis = 0
            for j in range(i + 1):
                yis += J[s][j]
            yis /= Y[s]
            ys.append(yis)
        y.append(ys)

    G = []
    for s in range(S):
        if B[s] == 0:
            G_tmp = float('inf')
        else:
            G_tmp = 0
            for i in range(B[s] - 1):
                G_tmp += y[s][i]
            G_tmp = 1 - (G_tmp * 2 + 1) / B[s]
        G.append(G_tmp)

    I = []
    for s in range(S):
        if B[s] == 0:
            min = 0
        else:
            L = B[s]
            if N < L:
                L = N
            if G[s] == 0:
                a = B[s]
            else:
                a = ceil(1 / G[s]) + ceil((B[s] - ceil(1 / G[s])) * L / B[s])
            if a < B[s]:
                min = a
            else:
                min = B[s]
            if N < min:
                min = N
        I.append(min)
    return I

def gc_adaptive_GA(paras, iternum, population):
    Ger_max = population
    Epoch = iternum
    Pc = 0.8
    Pm = 0.3
    U = paras.U
    S = paras.S
    dna = init_ger(Ger_max, paras)
    I = get_server_bound(paras)
    #画图
    x_data = []
    y_data = []
    for epoch in range(Epoch):
        #画图
        x_data.append(epoch)

        Fittness = np.zeros(dna.shape[0])
        # 计算适应度
        for idv in range(dna.shape[0]):
            X = dna[idv]
            illegal = 0
            # 检测合法性
            for s in range(S):
                if np.sum(X[:, s]) >= I[s]:
                    illegal = 1
                    break
            J_tmp = get_J(X, paras)
            if J_tmp >= 0:
                Fittness[idv] = (1 - illegal * 0.9) * J_tmp
            else:
                Fittness[idv] = -10

        # 自然选择
        dna, best_idv, delta = survival_of_fit(dna, Fittness, Ger_max)
        # 画图
        J_plt = get_J(dna[best_idv], paras)
        y_data.append(J_plt)

        # 自适应改变遗传算子
        if epoch >= 1:
            Pc *= 1 + delta
            Pm *= 1 + delta

        if epoch == Epoch - 1:
            return dna[best_idv], x_data, y_data
        # 交叉
        for idv in range(Ger_max):
            pc = random.random()
            if pc < Pc:
                cross_point = random.randint(0, U - 2)
                another_parent = idv
                while another_parent == idv:
                    another_parent = random.randint(0, Ger_max - 1)
                X1 = dna[idv]
                X2 = dna[another_parent]
                new_X1 = np.concatenate((X1[0:cross_point + 1, :], X2[cross_point + 1:U, :]), axis=0)
                new_X2 = np.concatenate((X2[0:cross_point + 1, :], X1[cross_point + 1:U, :]), axis=0)
                dna.append(new_X1)
                dna.append(new_X2)
        # 变异
        for idv in range(Ger_max):
            pm = random.random()
            if pm < Pm:
                mute_point = random.randint(0, U - 1)
                X = dna[idv]
                new_X = X.copy()
                s = random.randint(0, S - 1)
                new_X[mute_point, :] = 0
                new_X[mute_point, s] = 1
                dna.append(new_X)

        dna = np.array(dna)