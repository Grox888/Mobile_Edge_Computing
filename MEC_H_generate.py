import numpy as np
from math import ceil
from math import sqrt
from math import log
import random

#to generate physics location
def genLocation(U, S, gapofserver):
    xlength = (S**0.5 + 1) * (gapofserver)
    ylength = xlength
    index = 0
    serverMap = np.zeros((S, 2))
    if S != 2:
        for i in range(ceil(sqrt(S))):
            for j in range(ceil(sqrt(S))):
                serverMap[index, 0] = i*gapofserver
                serverMap[index, 1] = j*gapofserver
                index += 1
    else:
        for i in range(2):
            serverMap[index, 0] = gapofserver
            serverMap[index, 1] = (i + 1) * gapofserver
            index += 1
    index = 0
    userMap = np.zeros((U, 2))
    for i in range(U):
        userMap[index, 0] = xlength*random.random()
        userMap[index, 1] = ylength*random.random()
        index += 1
    return serverMap, userMap

#to generate H matrix
def genGain(U, S, N, gapofserver):
    serverMap, userMap = genLocation(U, S, gapofserver)
    H = np.zeros((U, S, N))
    for i in range(U):
        for j in range(S):
            dis = ((userMap[i, 0] - serverMap[j, 0])**2 + (userMap[i, 1] - serverMap[j, 1])**2)**0.5
            gain_DB = 140.7 + 36.7*log(dis/1000, 10)
            H[i, j, :] = 1/(10**(gain_DB/10))
    return H