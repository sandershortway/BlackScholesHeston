import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random as rd
import csv
from scipy.stats import norm
from scipy.integrate import quad

# Dynamic parameters
S0 = 100          # Spot price
v0 = 0.05       # Intial volatility
r = 0           # Interest rate
kappa = 10       # Mean-reversion rate of volatility
vbar = 0.05     # Long-term variance
rho = -0.9      # Correlation between BM's
zeta = .75        # Volatility of variance

# Option parameters
K = 100           # Strike price
T = 1          # Maturity

#Returns if Feller Condition is satisfied
def Feller(kappa, vbar, zeta):
  return 2 * kappa * vbar >= zeta ** 2

#Calculates the char f of log(S(t)) in w
def HestonCharf(S0, v0, vbar, kappa, zeta, r, rho, T, w):
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
  cf = lambda w: HestonCharf(S0, v0, vbar, kappa, zeta, r, rho, T, w)
  i1 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w - 1j)) / (1j * w * cf(- 1j)))
  I1 = quad(i1, 0, np.inf)
  Pi1 = 0.5 + I1[0]/np.pi 
  i2 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w)) / (1j * w))
  I2 = quad(i2, 0, np.inf)
  Pi2 = 0.5 + I2[0]/np.pi 
  print("Error:", I1[1], I2[1])
  return S0 * Pi1 - K * np.exp(-r*T) * Pi2

# Plots integrand for call option for various W
def CharPlot(a, b, step):
  cf = lambda w: HestonCharf(S0, v0, vbar, kappa, zeta, r, rho, T, w)
  i1 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w - 1j)) / (1j * w * cf(- 1j)))
  i2 = lambda w: np.real((np.exp(-1j * w * np.log(K)) * cf(w)) / (1j * w))
  W = np.arange(a+step,b,step)
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

print(EuroCall(S0, K, T, v0, vbar, kappa, zeta, r, rho))
CharPlot(0,100,0.1)