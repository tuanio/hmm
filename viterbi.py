import numpy as np
from hmmlearn import hmm
from data import *


def viterbi(O):
    O -= 1
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


# observation
O = np.array([1, 2, 1, 1, 2, 2, 2, 2, 3])

#
P, best_state = viterbi(O)
print(P, best_state)

# test using hmmlearn library
model_hmm = hmm.MultinomialHMM(n_components=2)
model_hmm.startprob_ = PI
model_hmm.transmat_ = A
model_hmm.emissionprob_ = B
llh, decoded = model_hmm.decode(O.reshape(-1, 1))
print(np.exp(llh), decoded)
