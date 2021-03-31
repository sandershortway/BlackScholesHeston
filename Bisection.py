import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random as rd
import csv
import BlackScholes
from scipy.stats import norm

rd.seed(123)
plt.style.use('ggplot')

# Import data from data/filenameT.txt
def readcsv(filename, T):
  A = []
  filename = "data/" + filename + str(T) + ".csv"
  print("Opening:", filename)
  with open(filename) as csv_file:
    f = csv.reader(csv_file, delimiter=',')
    for row in f:
      A.append([float(row[0]), int(T), float(row[1])])
    return A
    

# Calculates the implied volatility using the bisection method
def bisection(S0, K, T, r, p, a, b, imax):
  for i in range(0, imax):
    mid = (a + b)/2
    BS = BlackScholes.BSEuroCall(S0, K, T, r, mid)
    if (abs(BS - p) < 0.01):
      iter.append(i)
      return mid
    elif (BS < p):
      a = mid
    elif (BS >= p):
      b = mid

# Plot basics
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
l = []

# Parameters
S0 = 120.09 # Current stock price
r = 0       # Interest rate

# Bisection method parameters
a = 0       # Left bound of interval
b = 1       # Right bound of the interval
imax = 25   # Max number of iterations
iter = []   # Number of iterations for solving bisection method

# Import data
D = [2, 8, 15, 22, 29, 36]
for d in D:
  iv, T, K = []
  l.append("Maturity:" + str(d))
  A = readcsv("apple", int(d))
  for i in range(0, len(A)):
    iv.append(100 * float(bisection(S0, A[i][0], A[i][1], r, A[i][2], a, b, imax)))
    T.append(A[i][1])
    K.append(A[i][0])

#ax.set_xlabel('Maturity')
#ax.set_ylabel('Strike price')
#ax.set_zlabel('Implied volatility')
bin = []
for i in range(1, 26):
  bin.append(i)
plt.hist(iter, bin, color="b")
#plt.legend(l)
plt.show()