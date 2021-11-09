import numpy as np
from data import *


def sample(T: int = 10):
    O = []
    prev_q = np.random.choice(a=N, p=PI)
    for _ in range(T):
        O += [np.random.choice(a=M, p=B[prev_q])]
        prev_q = np.random.choice(a=N, p=A[prev_q])
    return list(map(lambda x: V[x], O))


print(sample(5))
