'''
bisection.py, part of BlackScholesImpliedVolatility of github.com/sandershortway.
created on March 15th, 2021: 12:17
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random as rd
import csv
from scipy.stats import norm

rd.seed(123)

# Import data from data/filenamenum.txt
def readcsv(filename, T):
  A = []

  filename = "data/" + filename + str(T) + ".csv"
  print("Opening:", filename)
  
  with open(filename) as csv_file:
    f = csv.reader(csv_file, delimiter=',')
    z = 0
    for row in f:
        if (z != 0):
          A.append([float(row[2]), int(T), float(row[6])])
        z += 1
    return A
    

# Calculates price of European Call via Black-Scholes Formula
def BlackScholes(S0, K, T, r, sigma):
  d1 = (np.log(S0/K) + T * (r + (sigma ** 2)/2))/(sigma * np.sqrt(T))
  d2 = (np.log(S0/K) + T * (r - (sigma ** 2)/2))/(sigma * np.sqrt(T))
  return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Calculates the implied volatility using the bisection method
def bisection(S0, K, T, r, p, a, b, imax):
  for i in range(0, imax):
    mid = (a + b)/2
    BS = BlackScholes(S0, K, T, r, mid)
    if (BS < p):
      a = mid
    elif (BS >= p):
      b = mid
    if (abs(BS - p) < 0.01):
      return mid

# Parameters
S0 = 120.09 # Current stock price
r = 0       # Interest rate
iv = []     # Implied volatility

# Bisection method parameters
a = 0       # Left bound of interval
b = 1       # Right bound of the interval
imax = 25   # Max number of iterations

# Import files
A = readcsv("apple", 2)
for i in range(0, len(A)):
  iv.append(float(bisection(S0, A[i][0], A[i][1], r, A[i][2], a, b, imax)))
  X.append(A[i][0])
  Y.append(A[i][2])
  A[i].pop()

print(X)
print(Y)

X, Y = np.meshgrid(X, Y)

print(len(X), X)
print(len(Y), Y)
print(iv)