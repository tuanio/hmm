import numpy as np

Q = ['H', 'Đ']
V = np.array([1, 2, 3])

N = len(Q)
M = len(V)

PI = np.array([0.65, 0.35])

A = np.array([[0.65, 0.35],
              [0.65, 0.35]])

B = np.array([[0.2, 0.35, 0.45],
              [0.75, 0.15, 0.1]])

# observation
observation = np.array([1, 2, 1, 1, 2, 2, 2, 2, 3])