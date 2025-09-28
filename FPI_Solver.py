# This function runs the fixed point iteration. Heurstically, I have noticed the program's FPI converges after 23 iterations, but you can always bump it (just that storage cost increases ever so).

import numpy as np
from functions import k, kPrime, V_delta_prime, convergenceTest

def fixed_point_iteration(x, delta, max_iter=25):
    gridSize = len(x)
    array = np.zeros((max_iter, gridSize))  # Will hold all u_n horizontally
    array[0] = 0  # Initial guess: zero everywhere

    # Precompute V'_delta(x) (only depends on x and delta)
    V_prime = np.zeros((gridSize))
    for z in range(gridSize):
        V_prime[z] = V_delta_prime(x[z], delta)

    h = x[1] - x[0]  # Mesh size
    kDelta = k(delta, delta) 

    u_i = np.zeros((gridSize))

    for i in range(1, max_iter):  # Iteration loop
        for j in range(gridSize):  # Grid loop
            integral = 0
            if j > 0:  # Ensures there is a previous value to integrate with
                u_prev = array[i - 1][:j]
                kPri_values = kPrime(x[:j] - x[j] + delta, delta)  # ✅ pass delta
                integral = h * 0.5 * (
                    u_prev[0] * kPri_values[0]
                    + 2 * np.sum(u_prev[1:-1] * kPri_values[1:-1])
                    + u_prev[-1] * kPri_values[-1]
                )

            u_i[j] = (V_prime[j] + integral) / kDelta

        # Store u_n and check for convergence
        array[i] = u_i
        if convergenceTest(array[i - 1], array[i], i, x):
            return array, i

    return array, i
