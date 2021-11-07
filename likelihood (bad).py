# Hot: 0
# Cold: 1

import numpy as np
from itertools import permutations

# start probabilty
PI = { 'H': 0.8, 'C': 0.2 }

# transition probability
A = { 'H': { 'H': 0.6, 'C': 0.4 },
     'C': { 'H': 0.5, 'C': 0.5 } }

# observation probability
# emission probability
B = { 'H': {1: 0.2, 2: 0.4, 3: 0.4},
    'C': {1: 0.5, 2: 0.4, 3: 0.1} }

O = [3, 1, 3]

N = 2 # số lượng hidden state
M = 3 # số lượng observation
T = len(O)

def return_hiddenstates(T):
    for mask in range(1 << T):
        curr_state = ['H', 'C']
        yield ''.join([curr_state[bool(mask & (1 << i))] for i in range(T)])

hidden_states = return_hiddenstates(T)

# mask & i == 1 -> Cold
# mask & i == 0 -> Hot
ans = 0 # P(O)
for i in range(1 << T):
    Q = next(hidden_states)
    foo = 1 # P(O|Q)
    for H, o in zip(enumerate(Q), O):
        i, h = H
        foo *= bool(i) * A[Q[i - 1]][Q[i]] * B[h][o] + bool(not i) * PI[h] * B[h][o]
    ans += foo
print(ans)
# isolated word recognition
