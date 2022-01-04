import numpy as np
from MEC_paras import Paras
from MEC_H_generate import genGain
from MEC_offload_GA import offload_GA
from matplotlib import pyplot as plt
import streamlit as st




#below are the testing codes:
B_input = st.number_input('服务器总带宽：', 1e7, 50e7, 20e7, 1e7)
fs_input = st.number_input('服务器总算力：', 1e9, 100e9, 20e9, 1e9)
du_input = st.number_input('用户卸载数据量：', 1024*8, 1024*1024*8, 420*1024*8, 1024*8)
cu_input = st.number_input('用户所需算量：', 1e9, 20e9, 1e9, 1e9)
Pu_input = st.number_input('用户功率：', 0.1, 1.0, 0.1, 0.1)
ful_input = st.number_input('用户本地算力：', 1e8, 50e8, 10e8, 1e8)
beta_t_input = st.number_input('用户beta_t:', 0.0, 1.0, 0.5, 0.1)
K_input = st.number_input('用户芯片系数：', 1e-27, 10e-27, 1e-27, 1e-27)
zigma_input = st.number_input('zigma:', 1e-6, 10e-6, 1e-6, 1e-6)
eps_input = st.number_input('误差控制eps：', 0.00005, 0.05, 0.05, 0.00005)
gapofserver_start = st.slider('MEC系统物理距离：', 10, 500, 50, 10)
U_start = st.slider('U起始：', 5, 30, 5)
S_start = st.slider('S起始：', 1, 10, 4)
N_start = st.slider('N起始：', 1, 10, 3)
loop = st.slider('循环数：', 1, 50, 10, 1)
step = st.slider('步长：', 1, 5, 1, 1)
iter_num_start = st.slider('GA算法迭代次数：', 10, 200, 100, 10)
population_start = st.slider('GA算法种群上限：', 10, 100, 50, 5)
Pc_start = st.slider('GA算法交叉概率：', 0.1, 0.9, 0.8, 0.1)
Pm_start = st.slider('GA算法变异概率：', 0.05, 0.95, 0.1, 0.05)
para_to_chg = st.radio('迭代参数：', ('U', 'S', 'N', 'GA迭代数', 'GA种群上限', 'GA交叉概率', 'GA变异概率'), index=0)


if st.button('提交'):
    flag = [0, 0, 0, 0, 0, 0, 0]
    if para_to_chg == 'U':
        flag[0] = 1
    if para_to_chg == 'S':
        flag[1] = 1
    if para_to_chg == 'N':
        flag[2] = 1
    if para_to_chg == 'GA迭代数':
        flag[3] = 1
    if para_to_chg == 'GA种群上限':
        flag[4] = 1
    if para_to_chg == 'GA交叉概率':
        flag[5] = 1
    if para_to_chg == 'GA变异概率':
        flag[6] = 1

    res_J = []
    res_U = []
    for idx in range(loop):
        # below are the object structures
        # user(du, cu, Pu, lamda, beta_t, ful, K)
        # MEC(S, N, B, fs)
        # H, zigma, eps

        iter_num = iter_num_start + idx*step*flag[3]
        population = population_start + idx*step*flag[4]
        Pc = Pc_start + idx*step*flag[5]
        Pm = Pm_start + idx*step*flag[6]
        #Topology of physics model
        gapofserver = gapofserver_start

        #parameters of MEC
        U = U_start + idx*step*flag[0]
        S = S_start + idx*step*flag[1]
        N = N_start + idx*step*flag[2]
        B = B_input

        #parameters of users
        fs = [fs_input for i in range(S)]
        du = [du_input for i in range(U)]
        cu = [cu_input for i in range(U)]
        Pu = [Pu_input for i in range(U)]
        lamda = [1 for i in range(U)]
        beta_t = [beta_t_input for i in range(U)]
        ful = [ful_input for i in range(U)]
        K = K_input

        #parameters of uplink
        H = genGain(U, S, N, gapofserver)
        zigma = zigma_input

        #parameter of error control
        eps = eps_input

        # structure of Paras:
        # paras = {
        #     u_list,
        #     U,
        #     mec = {S, N, W, fs},
        #     H,
        #     zigma,
        #     eps,
        #     I_usj,
        #     theta
        # }

        paras = Paras(du, cu, Pu, lamda, beta_t, ful, K, U, S, N, B, fs, H, zigma, eps)
        J_X, X = offload_GA(paras, iter_num, population, Pc, Pm)
        res_J.append(J_X)
        res_U.append(U)

    res_J = np.array(res_J)
    res_U = np.array(res_U)

    fig = plt.figure()
    plt.plot(res_U, res_J, color='r', marker='o')
    st.pyplot(fig)