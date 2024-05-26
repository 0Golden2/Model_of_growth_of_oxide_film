import numpy as np


C_in1 = [0 for i in range(0, 100)] + [1 for j in range(100, 101)]
C_in2 = [0 for k in range(0, 100)] + [0 for p in range(100, 101)]

for n in range(1, 3):
    with open(f"Initial distribution for {n}-th equation.txt", "w+", encoding='utf-8') as f:
        if n == 1:
            np.savetxt(f, np.array(C_in1).T, fmt='%2.5f')
        if n == 2:
            np.savetxt(f, np.array(C_in2).T, fmt='%2.5f')