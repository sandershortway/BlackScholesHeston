"""
Bisection Method for Implied Volatility Calculation.

This module provides functions for calculating implied volatility using
the bisection root-finding method, along with utility functions for
data handling and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import csv
from typing import List, Tuple, Optional, Union


def check_list_equality(list_a: List, list_b: List) -> bool:
    """
    Check if two lists contain the same elements (order independent).
    
    Parameters
    ----------
    list_a : list
        First list to compare
    list_b : list
        Second list to compare
        
    Returns
    -------
    bool
        True if lists contain the same elements, False otherwise
    """
    if len(list_a) != len(list_b):
        return False
    return sorted(list_a) == sorted(list_b)


def read_csv_data(filename: str, maturity: Union[int, float]) -> List[List[float]]:
    """
    Import option data from CSV file.
    
    Parameters
    ----------
    filename : str
        Base filename (without extension)
    maturity : int or float
        Maturity value to append to filename
        
    Returns
    -------
    list
        List of [strike, maturity, price] entries
        
    Raises
    ------
    FileNotFoundError
        If the specified file cannot be found
    """
    full_filename = f"data/{filename}{maturity}.csv"
    data = []
    
    try:
        with open(full_filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                data.append([float(row[0]), int(maturity), float(row[1])])
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {full_filename}")
    except (ValueError, IndexError) as e:
        raise ValueError(f"Error parsing CSV data: {e}")
    
    return data


def bisection_implied_volatility(s0: float, k: float, t: float, r: float, 
                                market_price: float, vol_min: float, 
                                vol_max: float, max_iterations: int,
                                pricing_function, tolerance: float = 0.02) -> Optional[float]:
    """
    Calculate implied volatility using the bisection method.
    
    Parameters
    ----------
    s0 : float
        Current stock price
    k : float
        Strike price
    t : float
        Time to maturity
    r : float
        Risk-free rate
    market_price : float
        Market price of the option
    vol_min : float
        Lower bound for volatility search
    vol_max : float
        Upper bound for volatility search
    max_iterations : int
        Maximum number of iterations
    pricing_function : callable
        Function to calculate theoretical option price
    tolerance : float, optional
        Convergence tolerance (default is 0.02)
        
    Returns
    -------
    float or None
        Implied volatility if found, None if not converged
    """
    for i in range(max_iterations):
        vol_mid = (vol_min + vol_max) / 2
        theoretical_price = pricing_function(s0, k, t, r, vol_mid)
        
        if abs(theoretical_price - market_price) < tolerance:
            return vol_mid
        elif theoretical_price < market_price:
            vol_min = vol_mid
        else:
            vol_max = vol_mid
    
    # If we reach here, the method didn't converge
    print(f"Warning: Bisection method did not converge after {max_iterations} iterations")
    return None


def plot_implied_volatility_surface(t_values: List[float], k_values: List[float], 
                                   price_matrix: List[List[float]], s0: float, 
                                   r: float, pricing_function,
                                   vol_bounds: Tuple[float, float] = (0.01, 2.0),
                                   max_iterations: int = 50) -> np.ndarray:
    """
    Create a 3D surface plot of implied volatilities.
    
    Parameters
    ----------
    t_values : list
        List of time to maturity values
    k_values : list
        List of strike price values
    price_matrix : list of lists
        2D matrix of option prices
    s0 : float
        Current stock price
    r : float
        Risk-free rate
    pricing_function : callable
        Function to calculate theoretical option price
    vol_bounds : tuple, optional
        (min_vol, max_vol) bounds for volatility search
    max_iterations : int, optional
        Maximum iterations for bisection method
        
    Returns
    -------
    np.ndarray
        2D array of implied volatilities
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 10))
    
    t_grid, k_grid = np.meshgrid(t_values, k_values)
    iv_matrix = []
    vol_min, vol_max = vol_bounds
    
    for i, k in enumerate(k_values):
        iv_row = []
        for j, t in enumerate(t_values):
            try:
                iv = bisection_implied_volatility(
                    s0, k, t, r, price_matrix[j][i], 
                    vol_min, vol_max, max_iterations, pricing_function
                )
                if iv is not None:
                    iv_row.append(100 * iv)  # Convert to percentage
                else:
                    iv_row.append(np.nan)
                    print(f"Failed to find IV for K={k}, T={t}, P={price_matrix[j][i]}")
            except Exception as e:
                print(f"Error calculating IV for K={k}, T={t}: {e}")
                iv_row.append(np.nan)
        
        iv_matrix.append(iv_row)
    
    iv_array = np.array(iv_matrix)
    
    # Only plot non-NaN values
    mask = ~np.isnan(iv_array)
    if np.any(mask):
        surf = ax.plot_surface(t_grid, k_grid, iv_array, cmap=cm.coolwarm)
        ax.view_init(15, 45)
        ax.set_xlabel('Maturity time: $T$')
        ax.set_ylabel('Strike price: $K$')
        ax.set_zlabel('Implied volatility (%)')
        ax.set_title('Implied Volatility Surface')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
    else:
        print("Warning: No valid implied volatilities calculated")
    
    plt.show()
    return iv_array


def plot_volatility_smile(k_values: List[float], s0: float, 
                         implied_volatilities: List[float],
                         title: str = "Volatility Smile") -> None:
    """
    Plot volatility smile showing implied volatility vs moneyness.
    
    Parameters
    ----------
    k_values : list
        List of strike prices
    s0 : float
        Current stock price
    implied_volatilities : list
        List of implied volatilities (in decimal form)
    title : str, optional
        Plot title
    """
    # Calculate moneyness
    moneyness = [100 * (k - s0) / s0 for k in k_values]
    iv_percent = [iv * 100 for iv in implied_volatilities if iv is not None]
    valid_moneyness = [m for m, iv in zip(moneyness, implied_volatilities) 
                      if iv is not None]
    
    plt.figure(figsize=(10, 6))
    plt.plot(valid_moneyness, iv_percent, 'bo-', linewidth=2, markersize=6)
    plt.xlabel("Moneyness (%)")
    plt.ylabel("Implied Volatility (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def calculate_implied_volatility_grid(s0: float, k_values: List[float], 
                                    t_values: List[float], r: float,
                                    price_matrix: List[List[float]], 
                                    pricing_function,
                                    vol_bounds: Tuple[float, float] = (0.01, 2.0),
                                    max_iterations: int = 1000) -> List[List[Optional[float]]]:
    """
    Calculate implied volatilities for a grid of strikes and maturities.
    
    Parameters
    ----------
    s0 : float
        Current stock price
    k_values : list
        List of strike prices
    t_values : list
        List of time to maturity values
    r : float
        Risk-free rate
    price_matrix : list of lists
        2D matrix of option prices [strike_index][time_index]
    pricing_function : callable
        Function to calculate theoretical option price
    vol_bounds : tuple, optional
        (min_vol, max_vol) bounds for volatility search
    max_iterations : int, optional
        Maximum iterations for bisection method
        
    Returns
    -------
    list of lists
        2D matrix of implied volatilities
    """
    vol_min, vol_max = vol_bounds
    iv_matrix = []
    
    for i, k in enumerate(k_values):
        iv_row = []
        for j, t in enumerate(t_values):
            try:
                # Convert time to years if needed (assuming monthly data)
                time_years = t / 12 if t > 2 else t
                iv = bisection_implied_volatility(
                    s0, k, time_years, r, price_matrix[i][j],
                    vol_min, vol_max, max_iterations, pricing_function
                )
                iv_row.append(iv)
            except Exception as e:
                print(f"Error calculating IV for K={k}, T={t}: {e}")
                iv_row.append(None)
        
        iv_matrix.append(iv_row)
    
    return iv_matrix


def main():
    """Main function for testing and demonstration."""
    # Example usage
    print("Bisection method module loaded successfully")
    
    # Test list equality function
    list1 = [1, 2, 3]
    list2 = [3, 1, 2]
    list3 = [1, 2, 4]
    
    print(f"Lists {list1} and {list2} are equal: {check_list_equality(list1, list2)}")
    print(f"Lists {list1} and {list3} are equal: {check_list_equality(list1, list3)}")


if __name__ == "__main__":
    main()


# Legacy function aliases for backward compatibility
def ListEquality(A, B):
    """Legacy alias for check_list_equality."""
    return check_list_equality(A, B)


def readcsv(filename, T):
    """Legacy alias for read_csv_data."""
    return read_csv_data(filename, T)


def bisection(S0, K, T, r, p, a, b, imax):
    """
    Legacy alias for bisection_implied_volatility.
    
    Note: This requires importing a Black-Scholes pricing function.
    """
    # Import here to avoid circular dependency
    try:
        from ..models.black_scholes import european_call_price
        return bisection_implied_volatility(
            S0, K, T, r, p, a, b, imax, european_call_price
        )
    except ImportError:
        # Fallback for legacy usage
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            import BlackScholes as bs
            return bisection_implied_volatility(
                S0, K, T, r, p, a, b, imax, bs.EuroCall
            )
        except ImportError:
            raise ImportError("Could not import Black-Scholes pricing function")


def SurfacePlot(T, K, P, S0, r):
    """Legacy alias for plot_implied_volatility_surface (requires pricing function)."""
    # This function requires a pricing function which creates dependencies
    raise NotImplementedError(
        "SurfacePlot requires a pricing function. "
        "Use plot_implied_volatility_surface with pricing_function parameter instead."
    )
