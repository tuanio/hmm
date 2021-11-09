import numpy as np
import copy
from data import *


def forward(A, B, PI, O):
    N = A.shape[0]
    T = O.shape[0]
    alpha = np.zeros(shape=(T, N))
    alpha[0, :] = PI * B[:, O[0]]
    for t in range(T - 1):
        for j in range(N):
            alpha[t + 1, j] = alpha[t, :].dot(A[:, j]) * B[j, O[t + 1]]

    termination = alpha[T - 1, :].sum()
    return termination, alpha


def backward(A, B, O):
    N = A.shape[0]
    T = O.shape[0]
    beta = np.zeros(shape=(T, N))
    beta[T - 1, :] = 1
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = A[i, :].dot(beta[t + 1, :]) * B[i, O[t + 1]]
    return beta


O = copy.deepcopy(observation) - 1

termination, alpha = forward(A, B, PI, O)
print(alpha)
print(termination)
