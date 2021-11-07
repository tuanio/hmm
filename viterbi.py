import numpy as np
from hmmlearn import hmm
from data import *

def viterbi(O):
    O -= 1
    T = len(O)
    delta = np.zeros(shape=(T, N))
    psi = np.zeros_like(delta)

    delta[0, :] = PI * B[:, O[0]]
    psi[0, :] = -1

    for t in range(1, T):
        for j in range(N):
            delta_max = delta[t - 1, :] * A[:, j]
            delta[t, j] = np.max(delta_max) * B[j, O[t]]
            psi[t, j] = np.argmax(delta_max)
    
    P_star = np.max(delta[T - 1, :])
    q = np.argmax(delta[T - 1, :]).astype('int')

    psi = psi.astype('int')

    best_state = [q]
    for t in range(T - 2, -1, -1):
        q = psi[t + 1, q]
        best_state = [q] + best_state
    return P_star, np.array(best_state)

O = np.array([1, 2, 1, 1, 2, 2, 2, 2, 3])
P, best_state = viterbi(O)
print(P, best_state)

model_hmm = hmm.MultinomialHMM(n_components=2)
model_hmm.startprob_ = PI
model_hmm.transmat_ = A
model_hmm.emissionprob_ = B
llh, decoded = model_hmm.decode(O.reshape(-1, 1))
print(np.exp(llh), decoded)