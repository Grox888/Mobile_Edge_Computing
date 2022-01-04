import numpy as np

class user():
    def __init__(self, du, cu, Pu, lamda, beta_t, ful, K):
        self.du = du
        self.cu = cu
        self.Pu = Pu
        self.lamda = lamda
        self.beta_t = beta_t
        self.ful = ful
        self.K = K

class MEC():
    def __init__(self, S, N, B, fs_list):
        self.S = S
        self.N = N
        self.W = B/N
        fs = []
        for idx in range(S):
            fs.append(fs_list[idx])
        self.fs = fs

class Paras():
    def __init__(self, du_list, cu_list, Pu_list, lamda_list, beta_t_list, ful_list, K, U, S, N, B, fs_list, H, zigma, eps):
        u_list = []
        for u in range(U):
            user_tmp = user(du_list[u], cu_list[u], Pu_list[u], lamda_list[u], beta_t_list[u], ful_list[u], K)
            u_list.append(user_tmp)
        self.u_list = u_list
        self.U = U
        mec = MEC(S, N, B, fs_list)
        self.mec = mec
        self.H = H
        self.zigma = zigma
        self.eps = eps
        phi = []
        psai = []

        for u in range(U):
            phi.append(lamda_list[u]
                       * beta_t_list[u]
                       * du_list[u]
                       * ful_list[u]
                       / (cu_list[u] * mec.W))
        phi = np.array(phi)

        for u in range(U):
            psai.append(lamda_list[u]
                        * (1 - beta_t_list[u])
                        * du_list[u]
                        / (K
                           * (ful_list[u] ** 2)
                           * cu_list[u]
                           * mec.W))
        psai = np.array(psai)

        self.phi = phi
        self.psai = psai

    # to calculate I matrix

    def update_I_usj(self, X):
        user_list = self.u_list
        mec = self.mec
        h_usj = self.H
        U = self.U
        S = mec.S
        N = mec.N
        I_usj = np.zeros((U, S, N))
        for s in range(S):
            for u in range(U):
                if sum(X[u, s, :]) == 1:
                    for j in range(N):
                        for w in range(S):
                            if w == s:
                                continue
                            for k in range(U):
                                I_usj[u, s, j] += X[k, w, j] * user_list[k].Pu * h_usj[k, w, j]
        self.I_usj = I_usj

    # to calculate theta
    def update_theta(self, X):
        self.update_I_usj(X)
        mec = self.mec
        h_usj = self.H
        zigma = self.zigma
        U = self.U
        S = mec.S
        N = mec.N
        theta = np.zeros((U, S))
        for u in range(U):
            for s in range(S):
                for j in range(N):
                    theta[u, s] += h_usj[u, s, j] / (self.I_usj[u, s, j] + zigma ** 2)
        self.theta = theta