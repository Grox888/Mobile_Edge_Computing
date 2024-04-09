import numpy as np
from genH import genGain

class paras():
    def __init__(self, d_list, c_list, kar, P_max, f_local_list, fs_list, zigma, B, alpha_t, N, gapofserver):
        self.d = d_list
        self.c = c_list
        self.f_local = f_local_list
        self.f_server = fs_list
        self.kar = kar
        self.P_max = P_max
        self.zigma = zigma
        self.B = B
        self.N = N
        self.W = B / N
        self.alpha_t = alpha_t
        self.U = len(d_list)
        self.S = len(fs_list)
        self.H = genGain(self.U, self.S, self.N, gapofserver) * 1e11
        self.get_localET()
        self.get_zeta_lmd_miu()
    def get_localET(self):
        E_list = []
        T_list = []
        for u in range(self.U):
            El = self.kar * self.f_local[u] * self.c[u]
            Tl = self.c[u] / self.f_local[u]
            E_list.append(El)
            T_list.append(Tl)
        self.E_local = E_list
        self.T_local = T_list

    def get_zeta_lmd_miu(self):
        eta = []
        zeta = []
        lmd = []
        miu = np.zeros((self.U, self.S))
        for u in range(self.U):
            zeta_tmp = self.d[u] * self.alpha_t / self.T_local[u]
            lmd_tmp = self.d[u] * (1 - self.alpha_t) / self.E_local[u]
            eta_tmp = self.c[u] * self.alpha_t / self.T_local[u]
            eta.append(eta_tmp)
            zeta.append(zeta_tmp)
            lmd.append(lmd_tmp)
            for s in range(self.S):
                miu_tmp = self.H[u, s] / self.zigma
                miu[u, s] = miu_tmp
        self.zeta = zeta
        self.lmd = lmd
        self.miu = miu
        self.eta = eta

