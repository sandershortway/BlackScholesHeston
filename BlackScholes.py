import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import random as rd
import BrownianMotion as bm
import csv
from scipy.stats import norm

# Calculates price of European call via Black-Scholes Formula
def EuroCall(S0, K, T, r, sigma):
  d1 = (np.log(S0/K) + T * (r + (sigma ** 2)/2))/(sigma * np.sqrt(T))
  d2 = (np.log(S0/K) + T * (r - (sigma ** 2)/2))/(sigma * np.sqrt(T))
  return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Calculates price of European put via put-call parity in Black-Scholes formula
def EuroPut(S0, K, T, r, sigma):
  call = BSEuroCall(S0, K, T, r, sigma)
  return call + K * np.exp(-r * T) - S0

# Surface plot for price of EuroCall for various strikes and maturities
def SurfacePlot(S0, r, sigma):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  P = []
  i = 0
  T = np.arange(1, 6, 0.5)
  K = np.arange(50, 155, 1)
  T, K = np.meshgrid(T, K)
  for Trow in T:
    P.append([])
    for t in Trow:
      P[i].append(EuroCall(S0, K[i][0], t, r/100, sigma))
    i += 1
  p = np.array(P) # Convert to numpy array
  surf = ax.plot_surface(T, K, p, cmap=cm.coolwarm)
  ax.view_init(15, 145)
  ax.set_xlabel('Maturity time: $T$')
  ax.set_ylabel('Strike price: $K$')
  ax.set_zlabel('Black Scholes price')
  #plt.title("Interest rate = " + str(r))
  fig.set_size_inches(10, 10)
  plt.savefig("fig/BlackScholesFormula" + str(r) + ".png")

'''
##### Work in Progress #################################
# Calculates price of European call via simulation
def EuroCallSim(S0, K, T, r, sigma):
  N = 1000 # number of samples
  n = round(T/dt)
  t = np.linspace(0, T, n)
  p = []
  for i in range(0, N):
    St = S0 * np.exp((r * T + sigma * np.random.normal(0,1))
    print(St)
    p.append(max(St - K, 0))
  return np.exp(-r * T) * np.mean(p)

# Calculates price of a forward contract via Black-Scholes Formula
def ForwCont(S0, K, T, r):
  return S0 - np.exp(-r * T) * K
########################################################
'''

if (__name__ == "__main__"):
  np.random.seed(68486)
  T = 3
  K = 110
  R = [0, 1, 5, 15]
  dt = 0.01
  S0 = 100
  sigma = 0.06
  for r in R:
    SurfacePlot(S0, r, sigma)