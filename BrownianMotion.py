import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.random.seed(75438945)

# Generates a random walk with n steps and up prob. p
def RandomWalk(n, p):
    RW = []
    pos = 0
    for i in range(0, n):
        r = np.random.random()
        RW.append(pos)
        if (r < p):
            pos -= 1
        elif (r >= p):
            pos += 1
    return RW

# Generates a Brownian Motion path on [0,T] using N steps
def BrownianMotion(T, N):
    dt = T/N
    W = np.random.normal(0, dt, N)
    W = np.cumsum(W)
    return W

# Generates a correlated Brownian Motion path on [0,T]
# where the correlation coefficient is rho, correlated with BM
def CorBrownianMotion(B1, rho, T, N):
    dt = round(T/N)
    C = []
    B2 = BrownianMotion(T, N)
    for i in range(0, N):
        C.append(rho * B1[i] + np.sqrt(1 - rho ** 2) * B2[i])
    return C

# Generates a Brownian Motion with drift path on [0,T] for N-parameters(mu, sigma)
def BrownianMotionDrift(T, mu, sigma, dt):
    n = round(T/dt)
    t = np.linspace(0, T, n)
    W = np.random.normal(mu * dt, sigma * np.sqrt(dt), n-1)
    W = np.insert(W, 0, 0)
    W = np.cumsum(W)
    return W

# Generates a Geometric Brownian Motion path on [0,T]
def GeometricBrownianMotion(S0, mu, sigma, T, dt):
    n = round(T/dt)
    t = np.linspace(0, T, n)
    D = BrownianMotionDrift(T, mu, sigma, dt)
    G = []
    for i in range(0,n):
        G.append(S0 * np.exp(D[i]))
    plt.plot(t, G)
    return G

if (__name__ == "__main__"):
    T = 100
    t = np.linspace(0, T)
    plt.plot(RandomWalk(T,1/2))
    plt.xlim(0,100)
    plt.xlabel("Steps: $n$")
    plt.ylabel("Random Walk: $S_n$")
    plt.show()