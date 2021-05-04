import numpy as np
import matplotlib.pyplot as plt
import Heston as h
import BlackScholes as bs
from scipy.optimize import Bounds, differential_evolution
import time
start_time = time.time()

# EuroCall(S0, K, T, v0, vbar, kappa, zeta, r, rho):

# The objective function that needs to be minimized
# K is array of length N, T is array of length M
# P is matrix of size NxM
# P[i][j] is price of option with strike K[i] and maturity T[i]
# theta is array with heston parameters
# Order of theta: kappa, vbar, zeta, v0, rho
def ObjFunc(theta, S0, r, K, T, P):
  sum = 0
  for i in range(0,len(K)):
    for j in range(0,len(T)):
      p = h.EuroCall(S0, K[i], T[j], theta[3], theta[1], theta[0], theta[2], r, theta[4])
      sum += (P[i][j] - p) ** 2
  return sum

# Generates sample data that, in theory, allows for perfect calibration 
def CreateP(theta, S0, r, K, T):
  P = []
  i = 0
  for k in K:
    P.append([])
    for t in T:
      P[i].append(h.EuroCall(S0, k, t, theta[3], theta[1], theta[0], theta[2], r, theta[4]))
    i += 1
  return P

def Iteration(xk, convergence):
  print("x =",xk)
  print((time.time() - start_time), "seconds\n")

if (__name__ == "__main__"):
  goal = [1, 0.1, 1, 0.1, -0.8]
  T = [1, 2, 3, 4, 5]
  K = [90, 95, 100, 105, 110]
  S0 = 100
  r = 0.05
  P = CreateP(goal, S0, r, K, T) 

  # Setting bounds
  imax = 25
  lb = [0,0,0,0,-1]
  ub = [2,0.5,2,0.5,-0.5]
  Bounds = Bounds(lb, ub)

  result = differential_evolution(ObjFunc, Bounds, args=(S0, r, K, T, P), disp=True, maxiter=imax, callback=Iteration)
  print(result)
  x = result.x
  print("\n==========================================")
  print("Goal theta:", goal)
  print("Found theta:", x)
  print("Heston prices calculated:", result.nfev * len(K) * len(T))
  print("--- %s seconds ---" % (time.time() - start_time))