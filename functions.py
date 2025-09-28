# Libraries that will be used
import numpy as np
import matplotlib.pyplot as plt

# Input all the functions and get the derivatives necessary for the formula
def k(x):
  return np.where(np.abs(x) <= delta, (3 / (2 * delta**3)) * x, 0.0)

def kPrime(x):
  return np.where(np.abs(x) <= delta, (3 / (2 * delta**3)), 0.0)

def u_0(x):
  return np.sin(x)

def u_0Prime(x):
  return np.cos(x)

def f(x, u_val):
  return (10 / delta**5 * np.sin(x) * ((delta**2 - 2) * np.sin(delta) + 2 * delta * np.cos(delta))
  - 10 / (3 * delta**2) * u_val)

def fdx(x, u_val):
  return  10 / delta**5 * np.cos(x) * ((delta**2 - 2) * np.sin(delta) + 2 * delta * np.cos(delta))

def fdy(x, u_val):
  return -10 / (3 * delta**2)

# Obtained from classical IVP
def answer(x):
  return np.sin(x)

# This block of code obtains $V'_\delta(x) $ for us, which was base on the explicit formula obtained in the paper.
from scipy.integrate import quad

def V_delta_prime(x, delta):
    # Defining some terms used in the explicit formula
    term1 = fdx(x - delta, u_0(x - delta))
    term2 = fdy(x - delta, u_0(x - delta)) * u_0Prime(x - delta)
    term3 = u_0(x - 2 * delta) * k(-delta)

    # Defining the first integral term (from x - 2*delta to x)
    def integrand1(y):
        return u_0Prime(x - delta) * k(y - x + delta)

    integral1, _ = quad(integrand1, x - 2 * delta, x)

    # Defining the second integral term (from x - 2*delta to 0)
    def integrand2(y):
        return u_0(y) * kPrime(y - x + delta)

    integral2, _ = quad(integrand2, x - 2 * delta, 0)

    # The final expression for V_delta_prime(x)
    V_prime = term1 + term2 + term3 + integral1 + integral2
    return V_prime

# This is my method for checking uniform convergence between iterations to know if we can reach terminate earlier.
def convergenceTest(u_prev, u_current, i, tolerance=1e-14, grid):
  # Checking for uniform convergence with infinity norm
  sup_difference = np.max(np.abs(u_current - u_prev))
  if sup_difference < tolerance:
    print(f"Uniform convergence in FPI established at iteration {i} with difference {sup_difference}.")
    # Quickly getting the infinity norm against the true solution
    sup_norm = np.max(np.abs(u_current - answer(grid)))
    print("Infinity Norm:", f"{sup_norm:.2e}")
    return True
