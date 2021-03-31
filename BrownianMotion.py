import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
np.random.seed(547915)

# Generates a Brownian Motion path on [0,T]
def BrownianMotion(T, dt):
    n = round(T/dt)
    t = np.linspace(0, T, n)
    W = [0]
    for i in range(1,n):
        inc = np.random.normal(0, np.sqrt(dt))
        W.append(W[i-1] + inc)
    plt.plot(t, W)
    return W

# Generates a Brownian Motion with drift path on [0,T] for N-parameters(mu, sigma)
def BrownianMotionDrift(T, mu, sigma, dt):
    n = round(T/dt)
    t = np.linspace(0, T, n)
    W = [0]
    for i in range(1,n):
        inc = np.random.normal(mu * dt, sigma ** 2)
        W.append(W[i-1] + inc)
    plt.plot(t, W)
    return W

# Generates a Geometric Brownian Motion path on [0,T]
def GeometricBrownianMotion(T, mu, sigma, dt):
    n = round(T/dt)
    t = np.linspace(0, T, n)
    W = [1]
    for i in range(1,n):
        inc = mu * dt + np.random.normal(0, np.sqrt(sigma))
        W.append(W[i-1] + inc)
    plt.plot(t, W)
    return W

T = 1
sigma = 1
dt = 0.01
mu = 1

#BrownianMotion(T, dt)
BrownianMotionDrift(T, mu, sigma, dt)
#GeometricBrownianMotion(T, mu, sigma, dt)
plt.xlabel("Time: $t$")
plt.ylabel("Brownian Motion: $B(t)$")
plt.show()