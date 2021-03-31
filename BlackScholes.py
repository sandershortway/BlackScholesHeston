import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random as rd
import csv
from scipy.stats import norm

# Calculates price of European Call via Black-Scholes Formula
def BSEuroCall(S0, K, T, r, sigma):
  d1 = (np.log(S0/K) + T * (r + (sigma ** 2)/2))/(sigma * np.sqrt(T))
  d2 = (np.log(S0/K) + T * (r - (sigma ** 2)/2))/(sigma * np.sqrt(T))
  return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Calculates price of European Call via put-call parity in Black-Scholes formula
def BSEuroPut(S0, K, T, r, sigma):
  call = BSEuroCall(S0, K, T, r, sigma)
  return call + K * np.exp(-r * T) - S0