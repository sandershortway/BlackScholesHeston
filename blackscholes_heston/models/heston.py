"""
Heston Stochastic Volatility Model Implementation.

This module provides functions for pricing European options using the Heston
stochastic volatility model, including characteristic function calculations,
path generation, and visualization tools.

References:
    Crisóstomo, 2014 - Characteristic function implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad


def check_feller_condition(kappa, v_bar, zeta):
    """
    Check if the Feller condition is satisfied for the Heston model.
    
    The Feller condition ensures that the variance process remains positive.
    Condition: 2 * kappa * v_bar >= zeta^2
    
    Parameters
    ----------
    kappa : float
        Mean reversion speed of variance
    v_bar : float
        Long-term variance level
    zeta : float
        Volatility of variance (vol of vol)
        
    Returns
    -------
    bool
        True if Feller condition is satisfied, False otherwise
    """
    return 2 * kappa * v_bar >= zeta ** 2


def characteristic_function(s0, v0, v_bar, kappa, zeta, r, rho, t, w):
    """
    Calculate the characteristic function of log(S(t)) in the Heston model.
    
    This implementation follows Crisóstomo (2014).
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    v0 : float
        Initial variance
    v_bar : float
        Long-term variance level
    kappa : float
        Mean reversion speed of variance
    zeta : float
        Volatility of variance
    r : float
        Risk-free rate
    rho : float
        Correlation between stock and variance Brownian motions
    t : float
        Time to maturity
    w : complex
        Frequency parameter
        
    Returns
    -------
    complex
        Characteristic function value
    """
    alpha = -((w ** 2) / 2.0) - ((1j * w) / 2.0)
    beta = kappa - rho * zeta * 1j * w
    gamma = (zeta ** 2) / 2.0
    h = np.sqrt(beta ** 2 - 4 * alpha * gamma)
    r_plus = (beta + h) / (zeta ** 2)
    r_min = (beta - h) / (zeta ** 2)
    g = r_min / r_plus
    
    c = kappa * (r_min * t - (2 / zeta**2) * 
                 np.log((1 - g * np.exp(-h * t)) / (1 - g)))
    d = r_min * ((1 - np.exp(-h * t)) / (1 - g * np.exp(-h * t)))
    
    return np.exp(c * v_bar + d * v0 + 1j * w * np.log(s0 * np.exp(r * t)))


def european_call_price(s0, k, t, v0, v_bar, kappa, zeta, r, rho):
    """
    Calculate European call option price using Heston model.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    k : float
        Strike price
    t : float
        Time to maturity
    v0 : float
        Initial variance
    v_bar : float
        Long-term variance level
    kappa : float
        Mean reversion speed of variance
    zeta : float
        Volatility of variance
    r : float
        Risk-free rate
    rho : float
        Correlation between stock and variance Brownian motions
        
    Returns
    -------
    float
        European call option price
    """
    def char_func(w):
        return characteristic_function(s0, v0, v_bar, kappa, zeta, r, rho, t, w)
    
    def integrand_1(w):
        return np.real((np.exp(-1j * w * np.log(k)) * char_func(w - 1j)) / 
                      (1j * w * char_func(-1j)))
    
    def integrand_2(w):
        return np.real((np.exp(-1j * w * np.log(k)) * char_func(w)) / 
                      (1j * w))
    
    integral_1 = quad(integrand_1, 0, np.inf)[0]
    pi_1 = 0.5 + integral_1 / np.pi
    
    integral_2 = quad(integrand_2, 0, np.inf)[0]
    pi_2 = 0.5 + integral_2 / np.pi
    
    return s0 * pi_1 - k * np.exp(-r * t) * pi_2


def calculate_pi2(s0, k, t, v0, v_bar, kappa, zeta, r, rho):
    """
    Calculate the Pi2 probability in the Heston model.
    
    This represents the risk-neutral probability that the option expires
    in-the-money.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    k : float
        Strike price
    t : float
        Time to maturity
    v0 : float
        Initial variance
    v_bar : float
        Long-term variance level
    kappa : float
        Mean reversion speed of variance
    zeta : float
        Volatility of variance
    r : float
        Risk-free rate
    rho : float
        Correlation between stock and variance Brownian motions
        
    Returns
    -------
    float
        Pi2 probability
    """
    def char_func(w):
        return characteristic_function(s0, v0, v_bar, kappa, zeta, r, rho, t, w)
    
    def integrand_2(w):
        return np.real((np.exp(-1j * w * np.log(k)) * char_func(w)) / 
                      (1j * w))
    
    integral_2 = quad(integrand_2, 0, np.inf)[0]
    return 0.5 + integral_2 / np.pi


def european_put_price(s0, k, t, v0, v_bar, kappa, zeta, r, rho):
    """
    Calculate European put option price using put-call parity.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    k : float
        Strike price
    t : float
        Time to maturity
    v0 : float
        Initial variance
    v_bar : float
        Long-term variance level
    kappa : float
        Mean reversion speed of variance
    zeta : float
        Volatility of variance
    r : float
        Risk-free rate
    rho : float
        Correlation between stock and variance Brownian motions
        
    Returns
    -------
    float
        European put option price
    """
    call_price = european_call_price(s0, k, t, v0, v_bar, kappa, zeta, r, rho)
    return call_price + k * np.exp(-r * t) - s0


def plot_characteristic_function(s0, k, t, v0, v_bar, kappa, zeta, r, rho,
                                a=0.1, b=10, step=0.1):
    """
    Plot the integrands for the characteristic function.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    k : float
        Strike price
    t : float
        Time to maturity
    v0 : float
        Initial variance
    v_bar : float
        Long-term variance level
    kappa : float
        Mean reversion speed of variance
    zeta : float
        Volatility of variance
    r : float
        Risk-free rate
    rho : float
        Correlation between stock and variance Brownian motions
    a : float, optional
        Lower bound for frequency range
    b : float, optional
        Upper bound for frequency range
    step : float, optional
        Step size for frequency grid
    """
    def char_func(w):
        return characteristic_function(s0, v0, v_bar, kappa, zeta, r, rho, t, w)
    
    def integrand_1(w):
        return np.real((np.exp(-1j * w * np.log(k)) * char_func(w - 1j)) / 
                      (1j * w * char_func(-1j)))
    
    def integrand_2(w):
        return np.real((np.exp(-1j * w * np.log(k)) * char_func(w)) / 
                      (1j * w))
    
    w_values = np.arange(a + step, b, step)  # Avoiding 0
    cf1_values = [integrand_1(w) for w in w_values]
    cf2_values = [integrand_2(w) for w in w_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(w_values, cf1_values, label=r"Integrand for $\Pi_1$")
    plt.plot(w_values, cf2_values, label=r"Integrand for $\Pi_2$")
    plt.legend()
    plt.grid(True)
    plt.xlim(a, b)
    plt.xlabel("Frequency (w)")
    plt.ylabel("Integrand Value")
    plt.title("Characteristic Function Integrands")
    plt.show()


def generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n):
    """
    Generate a sample path using the Heston stochastic volatility model.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    t : float
        Time horizon
    v0 : float
        Initial variance
    v_bar : float
        Long-term variance level
    kappa : float
        Mean reversion speed of variance
    zeta : float
        Volatility of variance
    r : float
        Risk-free rate
    rho : float
        Correlation between stock and variance Brownian motions
    n : int
        Number of time steps
        
    Returns
    -------
    np.ndarray
        Array of stock prices along the path
    """
    dt = float(t) / float(n)
    
    # Generate normal samples and correlated Brownian motions
    n1 = np.random.normal(0, 1, n + 1)
    n2 = np.random.normal(0, 1, n + 1)
    n2 = rho * n1 + np.sqrt(1 - rho ** 2) * n2  # Set correlation
    
    # Build Brownian motion paths
    w1 = [n1[0]]
    w2 = [n2[0]]
    for i in range(1, len(n1)):
        w1.append(w1[i-1] + np.sqrt(dt) * n1[i-1])
        w2.append(w2[i-1] + np.sqrt(dt) * n2[i-1])
    
    # Simulate variance path
    v = v0
    variance_path = []
    for i in range(0, n):
        variance_path.append(np.maximum(0, v))
        d_bt = w1[i+1] - w1[i]
        v += (kappa * (v_bar - np.maximum(v, 0)) * dt + 
              zeta * np.sqrt(np.maximum(v, 0)) * d_bt)
    
    # Simulate price path
    log_prices = []
    x = np.log(s0)
    for i in range(1, n):
        log_prices.append(x)
        d_bt = w2[i+1] - w2[i]
        x += ((r - 0.5 * variance_path[i]) * dt + 
              np.sqrt(variance_path[i]) * d_bt)
    
    return np.exp(log_prices)


def generate_heston_paths_with_variance(s0, t, v0, v_bar, kappa, zeta, r, rho, n):
    """
    Generate both stock price and variance paths using the Heston model.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    t : float
        Time horizon
    v0 : float
        Initial variance
    v_bar : float
        Long-term variance level
    kappa : float
        Mean reversion speed of variance
    zeta : float
        Volatility of variance
    r : float
        Risk-free rate
    rho : float
        Correlation between stock and variance Brownian motions
    n : int
        Number of time steps
        
    Returns
    -------
    tuple
        (variance_path, stock_price_path) as numpy arrays
    """
    dt = float(t) / float(n)
    
    # Generate normal samples and correlated Brownian motions
    n1 = np.random.normal(0, 1, n + 1)
    n2 = np.random.normal(0, 1, n + 1)
    n2 = rho * n1 + np.sqrt(1 - rho ** 2) * n2  # Set correlation
    
    # Build Brownian motion paths
    w1 = [n1[0]]
    w2 = [n2[0]]
    for i in range(1, len(n1)):
        w1.append(w1[i-1] + np.sqrt(dt) * n1[i-1])
        w2.append(w2[i-1] + np.sqrt(dt) * n2[i-1])
    
    # Simulate variance path
    v = v0
    variance_path = []
    for i in range(0, n):
        variance_path.append(np.maximum(0, v))
        d_bt = w1[i+1] - w1[i]
        v += (kappa * (v_bar - np.maximum(v, 0)) * dt + 
              zeta * np.sqrt(np.maximum(v, 0)) * d_bt)
    
    # Simulate price path
    log_prices = []
    x = np.log(s0)
    for i in range(1, n):
        log_prices.append(x)
        d_bt = w2[i+1] - w2[i]
        x += ((r - 0.5 * variance_path[i]) * dt + 
              np.sqrt(variance_path[i]) * d_bt)
    
    return np.array(variance_path), np.exp(log_prices)


def main():
    """Main function for testing and demonstration."""
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    # Heston parameters
    s0 = 100
    v0 = 0.15
    r = 0.05
    kappa = 1
    v_bar = 0.15
    rho = -0.5
    zeta = 0.5
    
    # Option parameters
    k = 100
    t = 1
    n = 1000  # Time steps
    
    # Calculate Pi2 probability
    pi2_value = calculate_pi2(s0, k, t, v0, v_bar, kappa, zeta, r, rho)
    print(f"Pi2 probability: {pi2_value}")
    
    # Generate and plot a sample path
    # np.random.seed(42)  # For reproducible results
    # heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n)
    # time_grid = np.linspace(0, t, len(heston_path))
    # plt.plot(time_grid, heston_path, color="b", label="Stock price")
    # plt.xlabel("Time (years)")
    # plt.ylabel("Price ($)")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()


# Legacy function aliases for backward compatibility
def Feller(kappa, vbar, zeta):
    """Legacy alias for check_feller_condition."""
    return check_feller_condition(kappa, vbar, zeta)


def CharFunc(S0, v0, vbar, kappa, zeta, r, rho, T, w):
    """Legacy alias for characteristic_function."""
    return characteristic_function(S0, v0, vbar, kappa, zeta, r, rho, T, w)


def EuroCall(S0, K, T, v0, vbar, kappa, zeta, r, rho):
    """Legacy alias for european_call_price."""
    return european_call_price(S0, K, T, v0, vbar, kappa, zeta, r, rho)


def Pi2(S0, K, T, v0, vbar, kappa, zeta, r, rho):
    """Legacy alias for calculate_pi2."""
    return calculate_pi2(S0, K, T, v0, vbar, kappa, zeta, r, rho)


def EuroPut(S0, K, T, v0, vbar, kappa, zeta, r, rho):
    """Legacy alias for european_put_price."""
    return european_put_price(S0, K, T, v0, vbar, kappa, zeta, r, rho)


def CharPlot(a, b, step):
    """Legacy alias for plot_characteristic_function (requires global parameters)."""
    # This function requires global parameters which is not ideal
    # Users should use plot_characteristic_function directly
    raise NotImplementedError(
        "CharPlot requires global parameters. Use plot_characteristic_function instead."
    )


def HestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for generate_heston_path."""
    return generate_heston_path(S0, T, v0, vbar, kappa, zeta, r, rho, N)


def PlotHestonPath(S0, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for generate_heston_paths_with_variance."""
    return generate_heston_paths_with_variance(S0, T, v0, vbar, kappa, zeta, r, rho, N)
