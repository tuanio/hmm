import numpy as np
import copy
from hmmlearn import hmm
from data import *

model_hmm = hmm.MultinomialHMM(n_components=2, n_iter=24, init_params='')
model_hmm.startprob_ = PI
model_hmm.transmat_ = A
model_hmm.emissionprob_ = B

O = copy.deepcopy(observation).reshape(-1, 1) - 1

print("Observation:", O[:, 0])

print("Problem 1: Evaluation")
score = np.exp(model_hmm.score(O))
print(score)

print("Problem 2: Decode")
llh, decoded = model_hmm.decode(O)
print(np.exp(llh), decoded)

print("Problem 3: Learn")
model_hmm.fit(O)
print("A:")
print(model_hmm.transmat_)
print("B:")
print(model_hmm.emissionprob_)
print("PI:")
print(model_hmm.startprob_)