import random
import numpy as np
from MEC import get_J

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

    # 计算自适应改变量
    min_1 = Fittness[Fittness_sort[0]]
    mean_1 = np.mean(Fittness)
    chosen = Fittness[Fittness_sort[L // 2]]
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

def adaptive_GA(paras, iternum, population):
    Ger_max = population
    Epoch = iternum
    Pc = 0.4
    Pm = 0.05
    U = paras.U
    S = paras.S
    N = paras.N
    dna = init_ger(Ger_max, paras)

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
                if np.sum(X[:, s]) > N:
                    illegal = 1
                    break
            J_tmp = get_J(X, paras)
            if J_tmp >= 0:
                Fittness[idv] = (1 - illegal) * J_tmp
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