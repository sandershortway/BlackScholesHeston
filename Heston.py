import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random as rd
import csv
from scipy.stats import norm

# Initial parameters
S0 = 100        # Spot price
V0 = 0.1        # Intial volatility
r = 0.03        # Interest rate
kappa = 1       # Mean-reversion rate of volatility
theta = 0.05    # Long-term variance
lamb = 0.575    # ?
rho = -0.5      # Correlation between BM's