import numpy as np
import copy
from data import *

def viterbi(O):
    T = len(O)
    delta = np.zeros(shape=(T, N))
    psi = np.zeros_like(delta, dtype='int')

    delta[0, :] = PI * B[:, O[0]]
    psi[0, :] = -1

    for t in range(1, T):
        for j in range(N):
            delta_max = delta[t - 1, :] * A[:, j]
            delta[t, j] = np.max(delta_max) * B[j, O[t]]
            psi[t, j] = np.argmax(delta_max)

    P = np.max(delta[T - 1, :])
    Q = np.zeros_like(O)
    Q[-1] = np.argmax(delta[T - 1, :]).astype('int')
    for t in range(T - 2, -1, -1):
        Q[t] = psi[t + 1, Q[-1]]
    return P, Q

O = copy.deepcopy(observation) - 1
#
P, best_state = viterbi(O)
print(P, best_state)

