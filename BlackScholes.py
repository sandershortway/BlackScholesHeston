import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import random as rd
import BrownianMotion as bm
import Bisection as bi
import csv
from scipy.stats import norm
import time
start_time = time.time()

# Calculates price of European call via Black-Scholes Formula
def EuroCall(S0, K, T, r, sigma):
  d1 = (np.log(S0/K) + T * (r + (sigma ** 2)/2))/(sigma * np.sqrt(T))
  d2 = (np.log(S0/K) + T * (r - (sigma ** 2)/2))/(sigma * np.sqrt(T))
  return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Calculates price of European put via put-call parity in Black-Scholes formula
def EuroPut(S0, K, T, r, sigma):
  call = EuroCall(S0, K, T, r, sigma)
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
      P[i].append(EuroCall(S0, K[i][0], t, r, sigma))
    i += 1
  p = np.array(P) # Convert to numpy array
  surf = ax.plot_surface(T, K, p, cmap=cm.coolwarm)
  ax.view_init(15, 145)
  ax.set_xlabel('Maturity time: $T$')
  ax.set_ylabel('Strike price: $K$')
  ax.set_zlabel('Black Scholes price ($)')
  plt.show()

# Surface plot for price of EuroCall for various strikes and maturities
def IVSurfacePlot(S0, r, T, K, P):
  # Bisection parameters
  a = 0
  b = 2
  imax = 50

  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  T, K = np.meshgrid(T, K)
  i = 0
  j = 0
  IV = []
  for Trow in T:
    IV.append([])
    for t in Trow:
      vol = 100 * bi.bisection(S0, K[i][0], t/12, r, P[i][j], a, b, imax)
      # print(t, K[i][0], P[i][j], "->", vol)
      IV[i].append(vol)
      j += 1
    j = 0
    i += 1
  iv = np.array(IV) # Convert to numpy array
  print(iv)
  surf = ax.plot_surface(T, K, iv, cmap=cm.coolwarm)
  ax.view_init(15, 55)
  ax.set_xlabel('Maturity time: $T$')
  ax.set_ylabel('Strike price: $K$')
  ax.set_zlabel('Implied Volatility (%)')
  plt.savefig("fig/IVSurfacePlot" + str(start_time) + ".pdf")
  plt.show()
  return IV

if (__name__ == "__main__"):
  np.random.seed(3)
  plt.style.use('ggplot')
  r = 0
  S0 = 133
  
  T = [1,2,3,4,5,6,7]
  K = []
  P = []
  M = []
  i = 0
  a = 0
  b = 2
  imax = 1000

  with open("data/AppleIV.csv", 'r') as csvfile:
      f = csv.reader(csvfile, delimiter=",")
      next(f) # Skips first header row
      for row in f:
        P.append([])
        Q = row[-7:]
        for q in Q:
          P[i].append(float(q))
        K.append(float(row[0]))
        i += 1
  j = 1
  for k in K:
    M.append(100 * (k - S0) / S0)
  # print(M)
  IV = IVSurfacePlot(S0,r,T,K,P)
  
  for j in range(1,8):
    IV = []
    for i in range(0, len(K)):
      vol = 100 * bi.bisection(S0, K[i], j/12, r, P[i][j-1], a, b, imax)
      # print(S0, K[i], j, r, P[i][j-1], a, b, imax, vol)
      IV.append(vol)
      # print(P[i][0], vol)
    if (j == 1):
      lbl = "Maturity: " + str(j) + " month"
    else:
      lbl = "Maturity: " + str(j) + " months"
    plt.plot(M, IV, label=lbl)
    plt.xlabel("Moneyness (%)")
    plt.ylabel("Implied volatility (%)")
    plt.title("Volatility Smile: $T = $" + str(j) + " months")
    plt.legend()
    plt.ylim(0, 120)
    plt.savefig("fig/" + str(j) + ".pdf")
    # plt.clf()
  plt.show()