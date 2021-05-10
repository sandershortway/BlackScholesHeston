import numpy as np
import matplotlib.pyplot as plt
import Heston as h
import BlackScholes as bs
from scipy.optimize import Bounds, differential_evolution
import time
import csv
start_time = time.time()

# The objective function that needs to be minimized
# K is array of length N, T is array of length M
# P is matrix of size NxM
# P[i][j] is price of option with strike K[i] and maturity T[i]
# theta is array with heston parameters
# Order of theta: kappa, vbar, zeta, v0, rho
def ObjFuncSqr(theta, S0, r, K, T, P):
  sum = 0
  for i in range(0,len(K)):
    for j in range(0,len(T)):
      p = h.EuroCall(S0, K[i], T[j], theta[3], theta[1], theta[0], theta[2], r, theta[4])
      sum += (P[i][j] - p) ** 2
  return sum

def ObjFuncAbs(theta, S0, r, K, T, P):
  sum = 0
  for i in range(0,len(K)):
    for j in range(0,len(T)):
      p = h.EuroCall(S0, K[i], T[j], theta[3], theta[1], theta[0], theta[2], r, theta[4])
      sum += np.abs(P[i][j] - p)
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

# Function being called after every iteration of DE
def Iteration(xk, convergence):
  print("x =",xk)
  print((time.time() - start_time), "seconds\n")

def ReadP(ticker):
  filename = "data/" + str(ticker) + ".csv"
  i = 0
  j = 0
  K = []
  P = []
  with open(filename) as csv_file:
    f = csv.reader(csv_file, delimiter=",")
    for row in f: 
      P.append([])
      for r in row:
        if (i == 0):
          K.append(float(r))
        else:
          P[j].append(float(r))
        i += 1
      i = 0
      j += 1
  return K, P

# Generates a random set of Heston parameters
def RandomHP(lb, ub, N):
  r = np.random.seed(start_time)
  Theta = []
  for i in range(N):
    theta = []
    for j in range(len(lb)):
      l = lb[j]
      r = ub[j]
      x = np.random.random()
      theta.append(round(l + x * (r-l),2))
    Theta.append(theta)
    print(theta)
  return Theta

if (__name__ == "__main__"):
  S0 = 100
  r = 0

  T = [1,2,3,4,5]
  K = [50,80,90,95,100,105,110,120,150]

  goal = [0.96, 0.72, 1.64, 0.86, -0.68]
  P = CreateP(goal, S0, r, K, T)

  # Setting bounds
  imax = 100
  lb = [0.1,0,0,0,-1]
  ub = [10,1,10,1,-0]

  # Run DE and print results
  print("Goal:", goal)
  result = differential_evolution(ObjFuncSqr, Bounds(lb,ub), args=(S0, r, K, T, P), disp=True, maxiter=imax, callback=Iteration)
  print(result)
  print("\n==========================================")
  print("Found theta:", result.x)
  print("Goal theta:", goal)
  print("Heston prices calculated:", result.nfev * len(K) * len(T))
  print("==========================================")
  print("Price matrix:\n", P)
  Px = CreateP(result.x, S0, r, K, T)
  print("Found price matrix:\n",Px)
  print("--- %s seconds ---" % (time.time() - start_time))