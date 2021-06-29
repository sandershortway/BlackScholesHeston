import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import random as rd
import csv
import BlackScholes as bs
from scipy.stats import norm

rd.seed(123)

# Checks if lists are equal
def ListEquality(A, B):
    if (len(A) != len(B)):
        return False
    return sorted(A) == sorted(B)

# Import data from data/filenameT.txt
# Returns array in the form: strike, maturity, price
def readcsv(filename, T):
  A = []
  filename = "data/" + filename + str(T) + ".csv"
  # print("Opening:", filename)
  with open(filename) as csv_file:
    f = csv.reader(csv_file, delimiter=',')
    for row in f:
      A.append([float(row[0]), int(T), float(row[1])])
    return A
    
# Calculates the implied volatility using the bisection method
def bisection(S0, K, T, r, p, a, b, imax):
  for i in range(0, imax):
    mid = (a + b)/2
    BS = bs.EuroCall(S0, K, T, r, mid)
    if (abs(BS - p) < 0.02):
      return mid
    elif (BS < p):
      a = mid
    elif (BS >= p):
      b = mid
  print("Nonetype error")

# Surface plot for implied vol of EuroCall for various strikes and maturities
def SurfacePlot(T, K, P, S0, r):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  IV = []
  i = 0
  T, K = np.meshgrid(T, K)
  for Trow in T:
    IV.append([])
    for j in range(0,len(Trow)):
      z = bisection(S0, K[i][j], T[i][j], r, P[j][i], 0, 1, 50)
      if (z is None):
        print("NoneType Error")
        print(K[i][j], T[i][j], P[j][i])
        return -1
      else:
        IV[i].append(100 * z)
    i += 1
  iv = np.array(IV) # Convert to numpy array
  surf = ax.plot_surface(T, K, iv, cmap=cm.coolwarm)
  ax.view_init(15, 45)
  ax.set_xlabel('Maturity time: $T$')
  ax.set_ylabel('Strike price: $K$')
  ax.set_zlabel('Implied volatility in %')
  fig.set_size_inches(10, 10)
  plt.show()