import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random as rd
import csv
from scipy.stats import norm
from scipy.integrate import quad
import time
start_time = time.time()

#Returns if Feller Condition is satisfied
def Feller(kappa, vbar, zeta):
  return 2 * kappa * vbar >= zeta ** 2

# Calculates the char f of log(S(t)) in w
# Function stems from Crisóstomo, 2014
def CharFunc(S0, v0, vbar, kappa, zeta, r, rho, T, w):
  alpha = - ((w ** 2) / 2.0) - ((1j * w) / 2.0)
  beta = kappa - rho * zeta * 1j * w
  gamma = (zeta ** 2) / 2.0
  h = np.sqrt(beta ** 2 - 4 * alpha * gamma)
  rplus = (beta + h) / (zeta ** 2)
  rmin = (beta - h) / (zeta ** 2)
  g = rmin / rplus
  C = kappa * (rmin * T - (2/zeta**2)*np.log( (1 - g * np.exp(-h*T))/(1-g) ))
  D = rmin * ( (1-np.exp(-h*T))/(1-g*np.exp(-h*T)) )
  return np.exp(C * vbar + D * v0 + 1j * w * np.log(S0 * np.exp(r*T)))

# Calculates price of European call option
def EuroCall(S0, K, T, v0, vbar, kappa, zeta, r, rho):
  cf = lambda w: CharFunc(S0, v0, vbar, kappa, zeta, r, rho, T, w)
  i1 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w - 1j)) / (1j * w * cf(- 1j)))
  I1 = quad(i1, 0, np.inf)
  Pi1 = 0.5 + I1[0]/np.pi 
  i2 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w)) / (1j * w))
  I2 = quad(i2, 0, np.inf)
  Pi2 = 0.5 + I2[0]/np.pi
  return S0 * Pi1 - K * np.exp(-r*T) * Pi2

# Calculates price of European put option
def EuroPut(S0, K, T, v0, vbar, kappa, zeta, r, rho):
  call = EuroCall(S0, K, T, v0, vbar, kappa, zeta, r, rho)
  return call + K * np.exp(-r * T) - S0

# Plots integrand for call option for various w
def CharPlot(a, b, step):
  cf = lambda w: CharFunc(S0, v0, vbar, kappa, zeta, r, rho, T, w)
  i1 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w - 1j)) / (1j * w * cf(- 1j)))
  i2 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w)) / (1j * w))
  W = np.arange(a+step,b,step) # avoiding 0
  CF1 = []
  CF2 = []
  for w in W:
    CF1.append(i1(w))
    CF2.append(i2(w))
  plt.plot(W, CF1)
  plt.plot(W, CF2)
  plt.legend(["Integrand for $\Pi_1$", "Integrand for $\Pi_2$"])
  plt.grid()
  plt.axis(xmin=a,xmax=b)
  plt.show()

# Euler scheme for volatility process
def PlotVarEuler(T, v0, vbar, kappa, zeta, rho, N):
  dt = 1/N
  time = np.arange(0, T, dt)
  var = []
  mean = []
  v = v0
  vplus = max(v, 0)
  N1 = np.random.normal(0, dt, N)
  for n in N1:
    var.append(v)
    mean.append(vbar)
    v += dt * kappa * (vbar - vplus) + n * zeta * np.sqrt(vplus)
    vplus = max(v, 0)
  plt.plot(var)
  plt.plot(mean)
  return var

def PlotSpotEuler(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
  dt = 1/N
  var = []
  spot = []
  v = v0
  vplus = max(v, 0)
  X = np.log(S0)
  N1 = np.random.normal(0, dt, N)
  N2 = np.random.normal(0, dt, N)
  N3 = rho * N1 + np.sqrt(1 -  rho ** 2) * N2 #CorrCoef(N1,N3)=rho
  for n in N1: # Generate variance process
    var.append(v)
    v += dt * kappa * (vbar - vplus) + n * zeta * np.sqrt(vplus)
    vplus = max(v, 0)
  
  for i in range(0,len(var)): # Generate spot price process
    spot.append(np.exp(X))
    X += (r - var[i]/2) * dt + np.sqrt(var[i]) * zeta * N3[i]
  plt.plot(spot)
  return spot

def HestonPath(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
  dt = float(T)/float(N)
  # Generate normal samples and correlated Brownian motions
  N1 = np.random.normal(0,1,N+1)
  N2 = np.random.normal(0,1,N+1)
  N2 = rho * N1 + np.sqrt(1 - rho ** 2) * N2 # Set CorrCoeff(N1,N2)=rho
  W1 = [N1[0]]
  W2 = [N2[0]]
  for i in range(1,len(N1)): 
    W1.append(W1[i-1] + np.sqrt(dt) * N1[i-1])
    W2.append(W2[i-1] + np.sqrt(dt) * N2[i-1])
  
  # Simulate Variance Path
  v = v0
  var = []
  for i in range(0,N):
    var.append(np.maximum(0,v))
    dBt = W1[i+1]-W1[i]
    v += kappa * (vbar - np.maximum(v,0)) * dt + zeta * np.sqrt(np.maximum(v,0)) * dBt
  
  # Simulate Price Path
  X = []
  x = np.log(S0)
  for i in range(1,N):
    X.append(x)
    dBt = W2[i+1]-W2[i]
    x += (r - 0.5 * var[i])*dt + zeta * np.sqrt(var[i]) * dBt
  
  S = np.exp(X)
  return S
  

if (__name__ == "__main__"):
  # Heston parameters
  S0 = 100        # Spot price
  v0 = 0.03        # Intial volatility
  r = 0.01           # Interest rate
  kappa = 0.5       # Mean-reversion rate of volatility
  vbar = 0.04    # Long-term variance
  rho = -0.9      # Correlation between BM's
  zeta = 1       # Volatility of variance
  P = []

  # Option parameters
  K = 100
  T = 1
  N = 10000 # Time steps
  M = 1000

  # Set Random Seed
  np.random.seed(645358)

  # Monte Carlo
  for j in range(M):
    HP = HestonPath(S0, K, T, v0, vbar, kappa, zeta, r, rho, N)
    P.append(np.maximum(HP[-1]-K,0))
  
  print("Price\t\t", EuroCall(S0, K, T, v0, vbar, kappa, zeta, r, rho))
  print("Monte Carlo\t", np.exp(-r*T)*np.mean(P))
  
  # plt.figaspect(16/9)
  # plt.grid()
  # plt.show()
  print("--- %s seconds ---" % (time.time() - start_time))