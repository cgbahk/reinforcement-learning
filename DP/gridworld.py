"""
David Silver's Reinforcement Learning - lecture 3
Dynamic programming
small grid world example

policy evaluation for 3 different methods
1. direct matrix operation
2. synchronous iteration
3. asynchronous iteration
"""
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle

# reward vector
R = np.ones(15) * -1
R[0] = 0

# discount factor
gam = 0.9

# state transition probability matrix
P = np.zeros([15, 15])
# P = np.full([15, 15], np.nan) # alternative
# setup P
P[0, 0] = 1
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]
for i in range(1, 15):
    x = i // 4  # from
    y = i % 4
    for d in range(0, 4):
        xx = x + dx[d]  # to
        yy = y + dy[d]
        if xx in range(0, 4) and yy in range(0, 4):
            ii = i + 4 * dx[d] + dy[d]
            P[i, ii % 15] = .25
        else:
            P[i, i] += .25


# method 1: direct matrix operation
A = np.eye(15) - gam * P
temp = np.linalg.solve(A[1:, 1:], R[1:])
v1 = np.zeros(15)
v1[1:] = temp

print(np.append(v1, 0).reshape(4, 4))  # pretty output


# method 2: synchronous iteration
v2 = np.zeros(15)
i = 0
while True:
    i += 1
    v2 = R + gam * P @ v2
    diff = np.linalg.norm(v2 - v1)
    if diff < 1e-3:
        break

print("iteration time: " + str(i))
print("error: " + str(np.linalg.norm(v2 - v1)))
print(np.append(v2, 0).reshape(4, 4))

eig = np.sort(abs(np.linalg.eigvals(P)))
plt.plot(eig, 'o')
plt.grid(True)
plt.show()

print(eig[-2])  # is too close to 1


# method 3: asynchronous iteration
v3 = np.zeros(15)
i = 0
while True:
    i += 1
    r = list(range(15))
    shuffle(r)
    for j in r:
        v3[j] = R[j] + gam * P[j] @ v3
    diff = np.linalg.norm(v3 - v1)
    if diff < 1e-3:
        break

print("iteration time: " + str(i))
print("error: " + str(np.linalg.norm(v3 - v1)))
print(np.append(v3, 0).reshape(4, 4))


# wild card
np.append(P[6], np.nan).reshape(4, 4)

# todo: make this to jupiter notebook
