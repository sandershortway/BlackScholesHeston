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
  I1 = quad(i1, 0, 10)
  Pi1 = 0.5 + I1[0]/np.pi 
  i2 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w)) / (1j * w))
  I2 = quad(i2, 0, 10)
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

def PlotSpotEuler(S0, K, T, v0, vbar, kappa, zeta, r, rho, mu, N):
  dt = 1/N
  vol = []
  spot = []
  v = v0
  vplus = max(v, 0)
  logS = np.log(S0)
  N1 = np.random.normal(0, dt, N)
  N2 = np.random.normal(0, dt, N)
  N3 = rho * N1 + np.sqrt(1 -  rho ** 2) * N2 #CorrCoef(N1,N3)=rho
  for i in range(0, N):
    vol.append(v)
    spot.append(np.exp(logS))
    v += dt * kappa * (vbar - vplus) + N1[i] * zeta * np.sqrt(vplus)
    vplus = max(v, 0)
    logS += dt * (mu - v / 2) + N3[i] * np.sqrt(v)
  # plt.plot(spot)
  return spot

if (__name__ == "__main__"):
  # Dynamic parameters
  S0 = 100        # Spot price
  v0 = 0.1        # Intial volatility
  r = 0.03           # Interest rate
  kappa = 5       # Mean-reversion rate of volatility
  vbar = 0.10     # Long-term variance
  rho = -0.5      # Correlation between BM's
  zeta = 1       # Volatility of variance

  # Option parameters
  K = 110
  T = 10
  N = 1000


  np.random.seed(645358)

  E = 0

  for i in range(100):
    E = EuroCall(S0, K, T, v0, vbar, kappa, zeta, r, rho)

  print(E)

  # plt.figaspect(16/9)
  # plt.title("Thee variance processes")
  # plt.ylabel("Variance: $v(t)$")
  # plt.grid()
  # plt.show()
  print("--- %s seconds ---" % (time.time() - start_time))