import numpy as np

S = ['H', 'C']
V = np.array([1, 2, 3])

N = len(S)
M = len(V)

PI = np.array([0.8, 0.2])

A = np.array([[0.6, 0.4],
              [0.5, 0.5]])

B = np.array([[0.2, 0.4, 0.4],
              [0.5, 0.4, 0.1]])

# observation
observation = np.array([1, 2, 1, 1, 2, 2, 2, 2, 3])