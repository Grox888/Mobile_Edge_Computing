import random
import numpy as np
from MEC import J

def offload_GA(paras, iter_num, population, Pc, Pm):
    Pu = []
    for user in paras.u_list:
        Pu.append(user.Pu)
    mec = paras.mec
    U = paras.U
    S = mec.S
    N = mec.N

    M = population
    ger = iter_num
    L = U
    pc = Pc
    pm = Pm
    dna = np.zeros((M, L, 3)) - 1
    #to generate the original population
    for idv in range(M):
        new_idv = np.zeros((L, 3)) - 1
        flag = True
        while flag == True:
            flag = False
            for u in range(L):
                new_idv[u, 0] = random.randint(0, S - 1)
                new_idv[u, 1] = random.randint(-1, N - 1)
                new_idv[u, 2] = Pu[u]*random.random()
            X_tmp = np.zeros((U, S, N))
            for u in range(L):
                j = int(new_idv[u, 1])
                if j != -1:
                    s = int(new_idv[u, 0])
                    X_tmp[u, s, j] = 1
            flag1 = False
            for u in range(U):
                for s in range(S):
                    if sum(X_tmp[u, s, :]) > 1:
                        flag1 = True
                        break
                if flag1 == True:
                    break
            flag2 = False
            for s in range(S):
                for j in range(N):
                    if sum(X_tmp[:, s, j]) > 1:
                        flag2 = True
                        break
                if flag2 == True:
                    break
            if flag1 or flag2:
                flag = True
        dna[idv, :, :] = new_idv.copy()

    J_x = 0
    x1 = np.zeros((M, L, 3)) - 1
    x2 = x1.copy()
    for epoch in range(ger):
        for i in range(M):
            if random.random() < pc:
                d = random.randint(0, M - 1)
                m = dna[d, :, :].copy()
                d = random.randint(0, L - 2)
                x1[i, :d + 1, :] = dna[i, :d + 1, :].copy()
                x1[i, d + 1:, :] = m[d + 1:, :].copy()
                x2[i, :d + 1, :] = m[:d + 1, :].copy()
                x2[i, d + 1:, :] = dna[i, d + 1:, :].copy()
        x3 = dna.copy()
        for i in range(M):
            if random.random() < pm:
                d = random.randint(0, L - 1)
                x3[i, d, 0] = random.randint(0, S - 1)
                x3[i, d, 1] = random.randint(-1, N - 1)
                x3[i, d, 2] = Pu[d]*random.random()
        new_dna = np.concatenate([dna, x1, x2, x3] ,axis=0)
        fi = np.zeros(4*M)
        for i in range(4*M):
            X_tmp = np.zeros((U, S, N))
            for u in range(L):
                s = int(new_dna[i, u, 0])
                j = int(new_dna[i, u, 1])
                if j != -1:
                    X_tmp[u, s, j] = 1
            flag1 = False
            for u in range(U):
                for s in range(S):
                    if sum(X_tmp[u, s, :]) > 1:
                        flag1 = True
                        break
                if flag1 == True:
                    break
            flag2 = False
            for s in range(S):
                for j in range(N):
                    if sum(X_tmp[:, s, j]) > 1:
                        flag2 = True
                        break
                if flag2 == True:
                    break
            if flag1 == True or flag2 == True:
                fi[i] = 0
            else:
                paras.update_theta(X_tmp)
                P = new_dna[i, :, 2]
                fi[i] = J(X_tmp, P, paras)
        idx_sort = np.argsort(-fi)
        while idx_sort.shape[0] > M:
            d = random.randint(0, idx_sort.shape[0] - 1)
            if random.random() < (d - 1)/idx_sort.shape[0]:
                idx_sort = np.delete(idx_sort, d, 0)
        for i in range(M):
            idv_keep_idx = idx_sort[i]
            dna[i, :, :] = new_dna[idv_keep_idx, :, :].copy()
        J_x = fi[idx_sort[0]]

    best_idv = dna[0]
    X = np.zeros((U, S, N), dtype=int)
    for u in range(U):
        for s in range(S):
            j = int(best_idv[u, 1])
            if j != -1:
                X[u, s, j] = 1
    return J_x, X






