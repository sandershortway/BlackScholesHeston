import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random as rd
import csv
from scipy.stats import norm
from scipy.integrate import quad
import time
start_time = time.time()
plt.style.use('ggplot')

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

# Calculates price of European call option
def Pi2(S0, K, T, v0, vbar, kappa, zeta, r, rho):
  cf = lambda w: CharFunc(S0, v0, vbar, kappa, zeta, r, rho, T, w)
  i1 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w - 1j)) / (1j * w * cf(- 1j)))
  I1 = quad(i1, 0, np.inf)
  Pi1 = 0.5 + I1[0]/np.pi 
  i2 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w)) / (1j * w))
  I2 = quad(i2, 0, np.inf)
  Pi2 = 0.5 + I2[0]/np.pi
  return Pi2

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

# Generates a Heston price path
def HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N):
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
  return np.exp(X)
  
# Generates a Heston price path
def PlotHestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N):
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
  return [var, np.exp(X)]

if (__name__ == "__main__"):
  plt.figure(figsize=(10,6))
  # Heston parameters
  S0 = 100 
  v0 = 0.15
  r = 0.05
  kappa = 1
  vbar = 0.15
  rho = -0.5
  zeta = 0.5
  P = []

  # Option parameters
  K = 0.01
  T = 5
  N = 1000 # Time steps
  M = 1000

  # Set Random Seed
  np.random.seed(round(start_time))
  print(Pi2(S0, K, T, v0, vbar, kappa, zeta, r, rho))
  # HP = HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N)
  # t = np.linspace(0, T, len(HP))
  # plt.plot(t, HP, color="b", label="Stock price")

  # M = np.max(HP)
  # m = np.min(HP)
  # plt.hlines(M, xmin=0, xmax=T, label="Maximum", color="r")
  # plt.hlines(m, xmin=0, xmax=T, label="Minimum", color="m")
  # plt.legend()

  # plt.xlabel("Time (years)")
  # plt.ylabel("Price ($)")
  # plt.ylim(50, 150)
  
  # plt.show()