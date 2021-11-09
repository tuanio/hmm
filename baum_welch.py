import numpy as np
from forward_backward import forward, backward
from data import *
import copy
from hmmlearn import hmm

PI = copy.deepcopy(PI)
A = copy.deepcopy(A)
B = copy.deepcopy(B)
O = copy.deepcopy(observation) - 1
n_iter = 25

model_hmm = hmm.MultinomialHMM(n_components=2, n_iter=n_iter, init_params='')
model_hmm.startprob_ = PI
model_hmm.transmat_ = A
model_hmm.emissionprob_ = B


def baum_welch(A, B, PI, O, n_iter=25):
    N = A.shape[0]
    T = O.shape[0]

    for _ in range(n_iter):
        termination, alpha = forward(A, B, PI, O)
        beta = backward(A, B, O)

        # calculate xi and gamma
        xi = np.zeros(shape=(T, N, N))
        for t in range(T - 1):
            for i in range(N):
                xi[t, i, :] = alpha[t, i] * A[i, :] * \
                    B[:, O[t + 1]] * beta[t + 1, :]
            xi[t, :, :] /= xi[t, :, :].sum()

        gamma = xi.sum(axis=2)
        gamma[T - 1, :] = alpha[T - 1, :] * beta[T - 1, :]
        gamma[T - 1, :] /= gamma[T - 1, :].sum()

        # re-estimate
        PI = gamma[0, :]
        A = xi[:T - 1, :, :].sum(axis=0) / gamma[:T - 1, :].sum(axis=0)

        gamma_sum = gamma.sum(axis=0)
        for v in V - 1:
            O_v = O == v
            B[:, v] = gamma[O_v, :].sum(axis=0) / gamma_sum
    return A, B, PI


A, B, PI = baum_welch(A, B, PI, O, n_iter=n_iter)
print(A)
print(B)
print(PI)

model_hmm.fit(O.reshape(-1, 1))

diff_A = np.linalg.norm(A - model_hmm.transmat_)
diff_B = np.linalg.norm(B - model_hmm.emissionprob_)
diff_PI = np.linalg.norm(PI - model_hmm.startprob_)
print(diff_A)
print(diff_B)
print(diff_PI)
print("A:")
print(model_hmm.transmat_)
print("B:")
print(model_hmm.emissionprob_)
print("PI:")
print(model_hmm.startprob_)