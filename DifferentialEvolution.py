import numpy as np
import matplotlib.pyplot as plt
import Heston as h
import BlackScholes as bs
from scipy.optimize import Bounds, differential_evolution, minimize
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

def MktObjFuncSqr(theta, S0, r):
  sum = 0
  with open("data/AppleYear.csv", 'r') as csvfile:
    f = csv.reader(csvfile, delimiter=",")
    next(f) # Skips first header row
    for row in f:
      p = h.EuroCall(S0, float(row[0]),float(row[1]),theta[3],theta[1],theta[0],theta[2],r,theta[4])
      sum += (p - float(row[2])) ** 2
  return sum

def MktObjFuncRelAbs(theta, S0, r):
  sum = 0
  with open("data/Appeltje.csv", 'r') as csvfile:
    f = csv.reader(csvfile, delimiter=",")
    next(f) # Skips first header row
    for row in f:
      p = h.EuroCall(S0, float(row[0]),float(row[1]),theta[3],theta[1],theta[0],theta[2],r,theta[4])
      sum += np.abs(p - float(row[2])) / float(row[2]) 
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
  S0 = 133.55
  r = -0.480

  # Setting bounds
  imax = 1000
  lb = [0.1,0,0.1,0,-1]
  ub = [10,1,10,1,0]

  K = []
  T = []
  P = []

  # Run DE and print results
  print("Stock price =", S0)
  result = differential_evolution(MktObjFuncRelAbs, Bounds(lb,ub), args=(S0, r), disp=True, maxiter=imax, callback=Iteration)
  print("\n==========================================")
  print(result)
  print("\n==========================================")
  theta = result.x
  with open("log/Appeltje.csv", 'w') as csvf:
    w = csv.writer(csvf, delimiter=",")
    w.writerow(["Strike","Maturity","Market Price","Calibration Price", "Abs error"])
    with open("data/Appeltje.csv", 'r') as csvfile:
      f = csv.reader(csvfile, delimiter=",")
      next(f) # Skips first header row
      for row in f:
        p = h.EuroCall(S0, float(row[0]),float(row[1]),theta[3],theta[1],theta[0],theta[2],r,theta[4])
        w.writerow([row[0],row[1], row[2], p, np.abs(p-float(row[2]))])
  print("--- %s seconds ---" % (time.time() - start_time))