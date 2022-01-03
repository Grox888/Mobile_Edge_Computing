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