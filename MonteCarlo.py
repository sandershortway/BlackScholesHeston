import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import random as rd
import csv
from scipy.stats import norm
from scipy.integrate import quad
import time
import Heston as h
import BlackScholes as bs
import BrownianMotion as bm
plt.style.use("ggplot")
start_time = time.time()
np.random.seed(round(start_time))

def ST(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  return HP[-1]

def AvgS(S0, T, v0, vbar, kappa, zeta, r, rho, N):
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  return np.average(HP)

# European options
def EuroCall(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  return np.maximum(HP[-1]-K,0)

def EuroPut(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  return np.maximum(K-HP[-1],0)

# Asian options
def AsianCallFloat(S0, T, v0, vbar, kappa, zeta, r, rho, N):
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  return np.maximum(HP[-1] - np.average(HP), 0)

def AsianPutFloat(S0, T, v0, vbar, kappa, zeta, r, rho, N):
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  return np.maximum(np.average(HP) - HP[-1], 0)

def AsianCallFix(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  return np.maximum(np.average(HP) - K, 0)

def AsianPutFix(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  return np.maximum(K - np.average(HP), 0)

# Barrier options
def BarrierDownIn(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N):
  if (S0 < B):
    return -1
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  if (np.min(HP) <= B):
    return np.maximum(HP[-1]-K,0)
  return 0

def BarrierDownOut(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N):
  if (S0 < B):
    return -1
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  if (np.min(HP) <= B):
    return 0
  return np.maximum(HP[-1]-K,0)

def BarrierUpIn(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N):
  if (S0 > B):
    return -1
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  if (np.max(HP) >= B):
    return np.maximum(HP[-1]-K,0)
  return 0

def BarrierUpOut(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N):
  if (S0 > B):
    return -1
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  if (np.max(HP) >= B):
    return 0
  return np.maximum(HP[-1]-K,0)

# Lookback options
def LookbackCall(S0, T, v0, vbar, kappa, zeta, r, rho, N):
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  return np.maximum(HP[-1] - np.min(HP), 0)

def LookbackPut(S0, T, v0, vbar, kappa, zeta, r, rho, N):
  HP = h.HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  return np.max(HP[-1] - np.max(HP), 0)

# Confidence Interval Calculations
def CI95(P, M):
  if (M > 1):
    s = 0
    Pm = np.mean(P)
    for p in P:
      s += (p - Pm) ** 2
    s = np.sqrt(s / M-1)
    
    UB = Pm + 1.96 * s / np.sqrt(M)
    LB = Pm - 1.96 * s / np.sqrt(M)
    return [LB, UB]

def CI99(P, M):
  if (M > 1):
    s = 0
    for p in P:
      s += (p - np.mean(P)) ** 2
    s = np.sqrt(s / M-1)
    
    UB = np.mean(P) + 2.58 * s / np.sqrt(M)
    LB = np.mean(P) - 2.58 * s / np.sqrt(M)
    return [LB, UB]

if (__name__ == "__main__"):
  # Heston parameters
  S0 = 100        # Spot price
  r = 0.02
  
  v0 = 0.05
  kappa = 1
  vbar = 0.05
  rho = -0.64
  zeta = 1
  T = [1,2,3,4,5]
  K = [50, 75, 85, 90, 95, 100, 105, 110, 115, 125, 150]
  N = 1000 # Time steps
  M = 100 # MC Samples
  P = []
  print(LookbackPut(S0, 1, v0, vbar, kappa, zeta, r, rho, N))
  # with open("data/ShortwayInc.csv", 'w') as csvf:
  #   w = csv.writer(csvf, delimiter=",")
  #   w.writerow(["Maturity", "Strike", "Heston price", "Simulation price", "95% CI", "99% CI"])
  #   for t in T:
  #     print("T =", t)
  #     for k in K:
  #       hp = h.EuroCall(S0, k, t, v0, vbar, kappa, zeta, r, rho)
  #       for i in range(M):
  #         P.append(EuroCall(S0, k, t, v0, vbar, kappa, zeta, r, rho, N))
  #       sim = np.mean(P)
  #       w.writerow([t, k, hp, sim,CI95(P, M), CI99(P, M)])
  #       P = []
  #       print("K =", k)
  #     print("\n")