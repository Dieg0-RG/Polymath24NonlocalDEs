import numpy as np
import matplotlib.pyplot as plt

from functions import answer
from fixed_point import fixed_point_iteration

# ---------------------------
# Parameters
# ---------------------------
delta = 10
n = 10000
grid = np.linspace(0, delta, n)

# ---------------------------
# Run fixed point iteration
# ---------------------------
h = grid[1] - grid[0]
print(f"Mesh size h = {h:.2e}")

u_n, fin_Iter = fixed_point_iteration(grid, delta)
print(f"Stopped after {fin_Iter} iterations.")

# ---------------------------
# Plot final solution vs exact
# ---------------------------
plt.figure(figsize=(10, 6))
plt.plot(grid, u_n[fin_Iter, :], label=f"u_{fin_Iter}")
plt.plot(grid, answer(grid), label="Exact Solution", color='black', linestyle='--')
plt.title("Graphs of u_final vs classical solution")
plt.xlabel("Grid Points")
plt.ylabel("u_n Values")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# Compute error (L2 norm)
# ---------------------------
# (NOTE: you had approx(x_point) but never defined approx)
# I assume you meant the computed solution at the final iteration
approx = u_n[fin_Iter, :]  

errors = answer(grid) - approx
rmseRES = np.sqrt(np.mean(errors**2))
print(f"L2 norm error: {rmseRES:.2e}")

# ---------------------------
# Plot evolution over iterations
# ---------------------------
indices = [1, 2, 3, 10, 15, 20, 21]

plt.figure(figsize=(10, 6))
for i in indices:
    plt.plot(grid, u_n[i, :], label=f"i = {i}")

plt.plot(grid, answer(grid), label="Exact Solution", color='black', linestyle='--')
plt.title("Graphs of u_n at Selected Indices")
plt.xlabel("Grid Points")
plt.ylabel("u_n Values")
plt.legend()
plt.grid(True)
plt.show()
