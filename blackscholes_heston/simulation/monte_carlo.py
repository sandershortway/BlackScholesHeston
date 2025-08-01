"""
Monte Carlo Simulation Methods for Option Pricing.

This module provides Monte Carlo simulation methods for pricing various
types of financial derivatives using the Heston stochastic volatility model,
including European, Asian, Barrier, and Lookback options.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import List, Tuple, Optional
from ..models.heston import generate_heston_path


def final_stock_price(s0: float, k: float, t: float, v0: float, v_bar: float,
                     kappa: float, zeta: float, r: float, rho: float, 
                     n_steps: int) -> float:
    """
    Get the final stock price from a Heston path simulation.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    k : float
        Strike price (not used in calculation, kept for compatibility)
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Final stock price
    """
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    return heston_path[-1]


def average_stock_price(s0: float, t: float, v0: float, v_bar: float,
                       kappa: float, zeta: float, r: float, rho: float, 
                       n_steps: int) -> float:
    """
    Calculate the average stock price from a Heston path simulation.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Average stock price along the path
    """
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    return np.average(heston_path)


# European Options
def european_call_payoff(s0: float, k: float, t: float, v0: float, v_bar: float,
                        kappa: float, zeta: float, r: float, rho: float, 
                        n_steps: int) -> float:
    """
    Calculate European call option payoff using Monte Carlo simulation.
    
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        European call option payoff
    """
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    return np.maximum(heston_path[-1] - k, 0)


def european_put_payoff(s0: float, k: float, t: float, v0: float, v_bar: float,
                       kappa: float, zeta: float, r: float, rho: float, 
                       n_steps: int) -> float:
    """
    Calculate European put option payoff using Monte Carlo simulation.
    
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        European put option payoff
    """
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    return np.maximum(k - heston_path[-1], 0)


# Asian Options
def asian_call_floating_strike_payoff(s0: float, t: float, v0: float, v_bar: float,
                                     kappa: float, zeta: float, r: float, rho: float, 
                                     n_steps: int) -> float:
    """
    Calculate Asian call option payoff with floating strike (average strike).
    
    Payoff = max(S_T - A, 0) where A is the average price
    
    Parameters
    ----------
    s0 : float
        Initial stock price
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Asian call option payoff with floating strike
    """
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    return np.maximum(heston_path[-1] - np.average(heston_path), 0)


def asian_put_floating_strike_payoff(s0: float, t: float, v0: float, v_bar: float,
                                    kappa: float, zeta: float, r: float, rho: float, 
                                    n_steps: int) -> float:
    """
    Calculate Asian put option payoff with floating strike (average strike).
    
    Payoff = max(A - S_T, 0) where A is the average price
    
    Parameters
    ----------
    s0 : float
        Initial stock price
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Asian put option payoff with floating strike
    """
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    return np.maximum(np.average(heston_path) - heston_path[-1], 0)


def asian_call_fixed_strike_payoff(s0: float, k: float, t: float, v0: float, v_bar: float,
                                  kappa: float, zeta: float, r: float, rho: float, 
                                  n_steps: int) -> float:
    """
    Calculate Asian call option payoff with fixed strike.
    
    Payoff = max(A - K, 0) where A is the average price
    
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Asian call option payoff with fixed strike
    """
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    return np.maximum(np.average(heston_path) - k, 0)


def asian_put_fixed_strike_payoff(s0: float, k: float, t: float, v0: float, v_bar: float,
                                 kappa: float, zeta: float, r: float, rho: float, 
                                 n_steps: int) -> float:
    """
    Calculate Asian put option payoff with fixed strike.
    
    Payoff = max(K - A, 0) where A is the average price
    
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Asian put option payoff with fixed strike
    """
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    return np.maximum(k - np.average(heston_path), 0)


# Barrier Options
def barrier_down_and_in_call_payoff(s0: float, k: float, t: float, barrier: float,
                                   v0: float, v_bar: float, kappa: float, zeta: float,
                                   r: float, rho: float, n_steps: int) -> float:
    """
    Calculate down-and-in barrier call option payoff.
    
    Option is activated if the stock price touches or goes below the barrier.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    k : float
        Strike price
    t : float
        Time to maturity
    barrier : float
        Barrier level
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Barrier option payoff, or -1 if invalid initial conditions
    """
    if s0 < barrier:
        return -1  # Invalid initial condition
    
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    
    if np.min(heston_path) <= barrier:
        return np.maximum(heston_path[-1] - k, 0)
    return 0


def barrier_down_and_out_call_payoff(s0: float, k: float, t: float, barrier: float,
                                    v0: float, v_bar: float, kappa: float, zeta: float,
                                    r: float, rho: float, n_steps: int) -> float:
    """
    Calculate down-and-out barrier call option payoff.
    
    Option is knocked out if the stock price touches or goes below the barrier.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    k : float
        Strike price
    t : float
        Time to maturity
    barrier : float
        Barrier level
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Barrier option payoff, or -1 if invalid initial conditions
    """
    if s0 < barrier:
        return -1  # Invalid initial condition
    
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    
    if np.min(heston_path) <= barrier:
        return 0  # Knocked out
    return np.maximum(heston_path[-1] - k, 0)


def barrier_up_and_in_call_payoff(s0: float, k: float, t: float, barrier: float,
                                 v0: float, v_bar: float, kappa: float, zeta: float,
                                 r: float, rho: float, n_steps: int) -> float:
    """
    Calculate up-and-in barrier call option payoff.
    
    Option is activated if the stock price touches or goes above the barrier.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    k : float
        Strike price
    t : float
        Time to maturity
    barrier : float
        Barrier level
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Barrier option payoff, or -1 if invalid initial conditions
    """
    if s0 > barrier:
        return -1  # Invalid initial condition
    
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    
    if np.max(heston_path) >= barrier:
        return np.maximum(heston_path[-1] - k, 0)
    return 0


def barrier_up_and_out_call_payoff(s0: float, k: float, t: float, barrier: float,
                                  v0: float, v_bar: float, kappa: float, zeta: float,
                                  r: float, rho: float, n_steps: int) -> float:
    """
    Calculate up-and-out barrier call option payoff.
    
    Option is knocked out if the stock price touches or goes above the barrier.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    k : float
        Strike price
    t : float
        Time to maturity
    barrier : float
        Barrier level
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Barrier option payoff, or -1 if invalid initial conditions
    """
    if s0 > barrier:
        return -1  # Invalid initial condition
    
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    
    if np.max(heston_path) >= barrier:
        return 0  # Knocked out
    return np.maximum(heston_path[-1] - k, 0)


# Lookback Options
def lookback_call_payoff(s0: float, t: float, v0: float, v_bar: float,
                        kappa: float, zeta: float, r: float, rho: float, 
                        n_steps: int) -> float:
    """
    Calculate lookback call option payoff.
    
    Payoff = max(S_T - min(S_t), 0) where min(S_t) is the minimum price along the path
    
    Parameters
    ----------
    s0 : float
        Initial stock price
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Lookback call option payoff
    """
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    return np.maximum(heston_path[-1] - np.min(heston_path), 0)


def lookback_put_payoff(s0: float, t: float, v0: float, v_bar: float,
                       kappa: float, zeta: float, r: float, rho: float, 
                       n_steps: int) -> float:
    """
    Calculate lookback put option payoff.
    
    Payoff = max(max(S_t) - S_T, 0) where max(S_t) is the maximum price along the path
    
    Parameters
    ----------
    s0 : float
        Initial stock price
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
    n_steps : int
        Number of time steps
        
    Returns
    -------
    float
        Lookback put option payoff
    """
    heston_path = generate_heston_path(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    return np.maximum(np.max(heston_path) - heston_path[-1], 0)


# Confidence Interval Calculations
def calculate_confidence_interval_95(prices: List[float], n_samples: int) -> Tuple[float, float]:
    """
    Calculate 95% confidence interval for Monte Carlo option prices.
    
    Parameters
    ----------
    prices : list
        List of option prices from Monte Carlo simulations
    n_samples : int
        Number of Monte Carlo samples
        
    Returns
    -------
    tuple
        (lower_bound, upper_bound) for 95% confidence interval
    """
    if n_samples <= 1:
        return (np.nan, np.nan)
    
    mean_price = np.mean(prices)
    
    # Calculate sample standard deviation
    variance = sum((p - mean_price) ** 2 for p in prices) / (n_samples - 1)
    std_dev = np.sqrt(variance)
    
    # 95% confidence interval (z = 1.96)
    margin_of_error = 1.96 * std_dev / np.sqrt(n_samples)
    lower_bound = mean_price - margin_of_error
    upper_bound = mean_price + margin_of_error
    
    return (lower_bound, upper_bound)


def calculate_confidence_interval_99(prices: List[float], n_samples: int) -> Tuple[float, float]:
    """
    Calculate 99% confidence interval for Monte Carlo option prices.
    
    Parameters
    ----------
    prices : list
        List of option prices from Monte Carlo simulations
    n_samples : int
        Number of Monte Carlo samples
        
    Returns
    -------
    tuple
        (lower_bound, upper_bound) for 99% confidence interval
    """
    if n_samples <= 1:
        return (np.nan, np.nan)
    
    mean_price = np.mean(prices)
    
    # Calculate sample standard deviation
    variance = sum((p - mean_price) ** 2 for p in prices) / (n_samples - 1)
    std_dev = np.sqrt(variance)
    
    # 99% confidence interval (z = 2.58)
    margin_of_error = 2.58 * std_dev / np.sqrt(n_samples)
    lower_bound = mean_price - margin_of_error
    upper_bound = mean_price + margin_of_error
    
    return (lower_bound, upper_bound)


def monte_carlo_option_pricing(option_payoff_func, n_simulations: int, 
                              discount_factor: float, *args, **kwargs) -> dict:
    """
    Generic Monte Carlo option pricing with confidence intervals.
    
    Parameters
    ----------
    option_payoff_func : callable
        Function that calculates option payoff
    n_simulations : int
        Number of Monte Carlo simulations
    discount_factor : float
        Discount factor (e^(-r*T))
    *args, **kwargs
        Arguments to pass to the payoff function
        
    Returns
    -------
    dict
        Dictionary containing price, confidence intervals, and statistics
    """
    payoffs = []
    
    for _ in range(n_simulations):
        payoff = option_payoff_func(*args, **kwargs)
        payoffs.append(payoff)
    
    # Calculate discounted prices
    discounted_payoffs = [p * discount_factor for p in payoffs]
    
    mean_price = np.mean(discounted_payoffs)
    ci_95 = calculate_confidence_interval_95(discounted_payoffs, n_simulations)
    ci_99 = calculate_confidence_interval_99(discounted_payoffs, n_simulations)
    
    return {
        'price': mean_price,
        'std_error': np.std(discounted_payoffs) / np.sqrt(n_simulations),
        'confidence_interval_95': ci_95,
        'confidence_interval_99': ci_99,
        'payoffs': discounted_payoffs
    }


def main():
    """Main function for testing and demonstration."""
    plt.style.use("ggplot")
    
    # Heston parameters
    s0 = 100        # Spot price
    r = 0.02
    v0 = 0.05
    kappa = 1
    v_bar = 0.05
    rho = -0.64
    zeta = 1
    
    # Option parameters
    k = 100
    t = 1
    n_steps = 1000  # Time steps
    
    # Test lookback put
    lookback_put_price = lookback_put_payoff(s0, t, v0, v_bar, kappa, zeta, r, rho, n_steps)
    print(f"Lookback Put Payoff: {lookback_put_price}")
    
    # Example Monte Carlo pricing
    n_simulations = 1000
    discount_factor = np.exp(-r * t)
    
    european_call_results = monte_carlo_option_pricing(
        european_call_payoff, n_simulations, discount_factor,
        s0, k, t, v0, v_bar, kappa, zeta, r, rho, n_steps
    )
    
    print(f"European Call Price: {european_call_results['price']:.4f}")
    print(f"95% CI: {european_call_results['confidence_interval_95']}")


if __name__ == "__main__":
    main()


# Legacy function aliases for backward compatibility
def ST(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for final_stock_price."""
    return final_stock_price(S0, K, T, v0, vbar, kappa, zeta, r, rho, N)


def AvgS(S0, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for average_stock_price."""
    return average_stock_price(S0, T, v0, vbar, kappa, zeta, r, rho, N)


def EuroCall(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for european_call_payoff."""
    return european_call_payoff(S0, K, T, v0, vbar, kappa, zeta, r, rho, N)


def EuroPut(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for european_put_payoff."""
    return european_put_payoff(S0, K, T, v0, vbar, kappa, zeta, r, rho, N)


def AsianCallFloat(S0, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for asian_call_floating_strike_payoff."""
    return asian_call_floating_strike_payoff(S0, T, v0, vbar, kappa, zeta, r, rho, N)


def AsianPutFloat(S0, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for asian_put_floating_strike_payoff."""
    return asian_put_floating_strike_payoff(S0, T, v0, vbar, kappa, zeta, r, rho, N)


def AsianCallFix(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for asian_call_fixed_strike_payoff."""
    return asian_call_fixed_strike_payoff(S0, K, T, v0, vbar, kappa, zeta, r, rho, N)


def AsianPutFix(S0, K, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for asian_put_fixed_strike_payoff."""
    return asian_put_fixed_strike_payoff(S0, K, T, v0, vbar, kappa, zeta, r, rho, N)


def BarrierDownIn(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for barrier_down_and_in_call_payoff."""
    return barrier_down_and_in_call_payoff(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N)


def BarrierDownOut(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for barrier_down_and_out_call_payoff."""
    return barrier_down_and_out_call_payoff(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N)


def BarrierUpIn(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for barrier_up_and_in_call_payoff."""
    return barrier_up_and_in_call_payoff(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N)


def BarrierUpOut(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for barrier_up_and_out_call_payoff."""
    return barrier_up_and_out_call_payoff(S0, K, T, B, v0, vbar, kappa, zeta, r, rho, N)


def LookbackCall(S0, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for lookback_call_payoff."""
    return lookback_call_payoff(S0, T, v0, vbar, kappa, zeta, r, rho, N)


def LookbackPut(S0, T, v0, vbar, kappa, zeta, r, rho, N):
    """Legacy alias for lookback_put_payoff."""
    return lookback_put_payoff(S0, T, v0, vbar, kappa, zeta, r, rho, N)


def CI95(P, M):
    """Legacy alias for calculate_confidence_interval_95."""
    return calculate_confidence_interval_95(P, M)


def CI99(P, M):
    """Legacy alias for calculate_confidence_interval_99."""
    return calculate_confidence_interval_99(P, M)
