"""
Black-Scholes Model Implementation.

This module provides functions for pricing European options using the
Black-Scholes-Merton model, including surface plotting capabilities
for option prices and implied volatilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm


def european_call_price(s0, k, t, r, sigma):
    """
    Calculate European call option price using Black-Scholes formula.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    k : float
        Strike price
    t : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
        
    Returns
    -------
    float
        European call option price
    """
    d1 = (np.log(s0 / k) + t * (r + (sigma ** 2) / 2)) / (sigma * np.sqrt(t))
    d2 = (np.log(s0 / k) + t * (r - (sigma ** 2) / 2)) / (sigma * np.sqrt(t))
    return s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def european_put_price(s0, k, t, r, sigma):
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
    r : float
        Risk-free rate
    sigma : float
        Volatility
        
    Returns
    -------
    float
        European put option price
    """
    call_price = european_call_price(s0, k, t, r, sigma)
    return call_price + k * np.exp(-r * t) - s0


def plot_price_surface(s0, r, sigma, t_range=(1, 6), k_range=(50, 155),
                      t_step=0.5, k_step=1):
    """
    Create a 3D surface plot of European call option prices.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    r : float
        Risk-free rate
    sigma : float
        Volatility
    t_range : tuple, optional
        (min_time, max_time) for maturity range
    k_range : tuple, optional
        (min_strike, max_strike) for strike range
    t_step : float, optional
        Step size for time grid
    k_step : float, optional
        Step size for strike grid
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))
    
    t_values = np.arange(t_range[0], t_range[1], t_step)
    k_values = np.arange(k_range[0], k_range[1], k_step)
    t_grid, k_grid = np.meshgrid(t_values, k_values)
    
    # Calculate option prices for each combination
    price_matrix = np.zeros_like(t_grid)
    for i in range(len(k_values)):
        for j in range(len(t_values)):
            price_matrix[i, j] = european_call_price(
                s0, k_values[i], t_values[j], r, sigma
            )
    
    surf = ax.plot_surface(t_grid, k_grid, price_matrix, cmap=cm.coolwarm)
    ax.view_init(15, 145)
    ax.set_xlabel('Maturity time: $T$')
    ax.set_ylabel('Strike price: $K$')
    ax.set_zlabel('Black-Scholes price ($)')
    ax.set_title('European Call Option Price Surface')
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def plot_implied_volatility_surface(s0, r, t_values, k_values, price_matrix,
                                   bisection_func, max_iterations=50):
    """
    Create a 3D surface plot of implied volatilities.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    r : float
        Risk-free rate
    t_values : array_like
        Array of time to maturity values
    k_values : array_like
        Array of strike price values
    price_matrix : array_like
        2D array of option prices
    bisection_func : callable
        Bisection function for implied volatility calculation
    max_iterations : int, optional
        Maximum iterations for bisection method
    
    Returns
    -------
    np.ndarray
        2D array of implied volatilities
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))
    t_grid, k_grid = np.meshgrid(t_values, k_values)
    
    # Bisection parameters
    vol_min = 0.01  # 1% minimum volatility
    vol_max = 2.0   # 200% maximum volatility
    
    iv_matrix = np.zeros_like(price_matrix)
    
    for i in range(len(k_values)):
        for j in range(len(t_values)):
            try:
                # Convert time to years if needed (assuming monthly data)
                time_years = t_values[j] / 12 if t_values[j] > 2 else t_values[j]
                iv = bisection_func(
                    s0, k_values[i], time_years, r, price_matrix[i][j],
                    vol_min, vol_max, max_iterations
                )
                iv_matrix[i, j] = iv * 100  # Convert to percentage
            except (ValueError, TypeError):
                iv_matrix[i, j] = np.nan
    
    # Remove NaN values for plotting
    mask = ~np.isnan(iv_matrix)
    if np.any(mask):
        surf = ax.plot_surface(
            t_grid, k_grid, iv_matrix, cmap=cm.coolwarm, alpha=0.8
        )
        ax.view_init(15, 55)
        ax.set_xlabel('Maturity time: $T$')
        ax.set_ylabel('Strike price: $K$')
        ax.set_zlabel('Implied Volatility (%)')
        ax.set_title('Implied Volatility Surface')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
    return iv_matrix


def plot_volatility_smile(s0, r, k_values, price_matrix, t_values,
                         bisection_func, max_iterations=1000):
    """
    Plot volatility smile for different maturities.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    r : float
        Risk-free rate
    k_values : array_like
        Array of strike price values
    price_matrix : array_like
        2D array of option prices
    t_values : array_like
        Array of time to maturity values
    bisection_func : callable
        Bisection function for implied volatility calculation
    max_iterations : int, optional
        Maximum iterations for bisection method
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate moneyness
    moneyness = [100 * (k - s0) / s0 for k in k_values]
    
    # Bisection parameters
    vol_min = 0.01
    vol_max = 2.0
    
    for j, t in enumerate(t_values):
        iv_values = []
        
        for i, k in enumerate(k_values):
            try:
                # Convert time to years if needed
                time_years = t / 12 if t > 2 else t
                iv = bisection_func(
                    s0, k, time_years, r, price_matrix[i][j],
                    vol_min, vol_max, max_iterations
                )
                iv_values.append(iv * 100)  # Convert to percentage
            except (ValueError, TypeError):
                iv_values.append(np.nan)
        
        # Plot only non-NaN values
        valid_indices = ~np.isnan(iv_values)
        if np.any(valid_indices):
            valid_moneyness = np.array(moneyness)[valid_indices]
            valid_iv = np.array(iv_values)[valid_indices]
            
            if len(t_values) == 1 or t == 1:
                label = f"Maturity: {t} month"
            else:
                label = f"Maturity: {t} months"
            
            plt.plot(valid_moneyness, valid_iv, label=label, marker='o')
    
    plt.xlabel("Moneyness (%)")
    plt.ylabel("Implied volatility (%)")
    plt.title("Volatility Smile")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 120)
    plt.show()


def main():
    """Main function for testing and demonstration."""
    np.random.seed(3)
    plt.style.use('ggplot')
    
    # Example parameters
    s0 = 100
    k = 100
    t = 1
    r = 0.05
    sigma = 0.2
    
    # Calculate option prices
    call_price = european_call_price(s0, k, t, r, sigma)
    put_price = european_put_price(s0, k, t, r, sigma)
    
    print(f"European Call Price: ${call_price:.2f}")
    print(f"European Put Price: ${put_price:.2f}")
    
    # Verify put-call parity
    parity_check = call_price - put_price - (s0 - k * np.exp(-r * t))
    print(f"Put-Call Parity Check: {parity_check:.10f}")
    
    # Create price surface plot
    plot_price_surface(s0, r, sigma)


if __name__ == "__main__":
    main()


# Legacy function aliases for backward compatibility
def EuroCall(S0, K, T, r, sigma):
    """Legacy alias for european_call_price."""
    return european_call_price(S0, K, T, r, sigma)


def EuroPut(S0, K, T, r, sigma):
    """Legacy alias for european_put_price."""
    return european_put_price(S0, K, T, r, sigma)


def SurfacePlot(S0, r, sigma):
    """Legacy alias for plot_price_surface."""
    plot_price_surface(S0, r, sigma)


def IVSurfacePlot(S0, r, T, K, P):
    """Legacy alias for plot_implied_volatility_surface (requires bisection module)."""
    # This function requires the bisection module which creates a circular dependency
    # Users should import the bisection module separately and use the new functions
    raise NotImplementedError(
        "IVSurfacePlot requires bisection module. "
        "Use plot_implied_volatility_surface with bisection_func parameter instead."
    )
