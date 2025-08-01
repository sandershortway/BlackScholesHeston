"""
Differential Evolution for Heston Model Parameter Calibration.

This module provides functions for calibrating Heston model parameters
to market option prices using differential evolution optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from typing import List, Tuple, Optional, Dict, Any
from scipy.optimize import Bounds, differential_evolution, minimize
from ..models.heston import european_call_price


def squared_error_objective(theta: List[float], s0: float, r: float, 
                           k_values: List[float], t_values: List[float],
                           price_matrix: List[List[float]]) -> float:
    """
    Calculate squared error objective function for Heston parameter calibration.
    
    Parameters
    ----------
    theta : list
        Heston parameters [kappa, v_bar, zeta, v0, rho]
    s0 : float
        Initial stock price
    r : float
        Risk-free rate
    k_values : list
        Array of strike prices
    t_values : list
        Array of time to maturity values
    price_matrix : list of lists
        2D matrix of market option prices P[i][j] for strike K[i] and maturity T[j]
        
    Returns
    -------
    float
        Sum of squared errors between market and model prices
    """
    kappa, v_bar, zeta, v0, rho = theta
    total_error = 0
    
    for i, k in enumerate(k_values):
        for j, t in enumerate(t_values):
            try:
                model_price = european_call_price(s0, k, t, v0, v_bar, kappa, zeta, r, rho)
                market_price = price_matrix[i][j]
                total_error += (market_price - model_price) ** 2
            except Exception as e:
                # Penalize invalid parameter combinations
                total_error += 1e6
    
    return total_error


def absolute_error_objective(theta: List[float], s0: float, r: float,
                            k_values: List[float], t_values: List[float],
                            price_matrix: List[List[float]]) -> float:
    """
    Calculate absolute error objective function for Heston parameter calibration.
    
    Parameters
    ----------
    theta : list
        Heston parameters [kappa, v_bar, zeta, v0, rho]
    s0 : float
        Initial stock price
    r : float
        Risk-free rate
    k_values : list
        Array of strike prices
    t_values : list
        Array of time to maturity values
    price_matrix : list of lists
        2D matrix of market option prices P[i][j] for strike K[i] and maturity T[j]
        
    Returns
    -------
    float
        Sum of absolute errors between market and model prices
    """
    kappa, v_bar, zeta, v0, rho = theta
    total_error = 0
    
    for i, k in enumerate(k_values):
        for j, t in enumerate(t_values):
            try:
                model_price = european_call_price(s0, k, t, v0, v_bar, kappa, zeta, r, rho)
                market_price = price_matrix[i][j]
                total_error += abs(market_price - model_price)
            except Exception as e:
                # Penalize invalid parameter combinations
                total_error += 1e6
    
    return total_error


def market_data_squared_error_objective(theta: List[float], s0: float, r: float,
                                       csv_filename: str) -> float:
    """
    Calculate squared error objective function using market data from CSV file.
    
    Parameters
    ----------
    theta : list
        Heston parameters [kappa, v_bar, zeta, v0, rho]
    s0 : float
        Initial stock price
    r : float
        Risk-free rate
    csv_filename : str
        Path to CSV file containing market data
        
    Returns
    -------
    float
        Sum of squared errors between market and model prices
    """
    kappa, v_bar, zeta, v0, rho = theta
    total_error = 0
    
    try:
        with open(csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader)  # Skip header row
            
            for row in reader:
                try:
                    k = float(row[0])
                    t = float(row[1])
                    market_price = float(row[2])
                    
                    model_price = european_call_price(s0, k, t, v0, v_bar, kappa, zeta, r, rho)
                    total_error += (model_price - market_price) ** 2
                except (ValueError, IndexError):
                    continue
                    
    except FileNotFoundError:
        print(f"Warning: Could not find file {csv_filename}")
        return 1e6
    
    return total_error


def market_data_relative_absolute_error_objective(theta: List[float], s0: float, r: float,
                                                 csv_filename: str) -> float:
    """
    Calculate relative absolute error objective function using market data from CSV file.
    
    Parameters
    ----------
    theta : list
        Heston parameters [kappa, v_bar, zeta, v0, rho]
    s0 : float
        Initial stock price
    r : float
        Risk-free rate
    csv_filename : str
        Path to CSV file containing market data
        
    Returns
    -------
    float
        Sum of relative absolute errors between market and model prices
    """
    kappa, v_bar, zeta, v0, rho = theta
    total_error = 0
    
    try:
        with open(csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader)  # Skip header row
            
            for row in reader:
                try:
                    k = float(row[0])
                    t = float(row[1])
                    market_price = float(row[2])
                    
                    if market_price > 0:  # Avoid division by zero
                        model_price = european_call_price(s0, k, t, v0, v_bar, kappa, zeta, r, rho)
                        total_error += abs(model_price - market_price) / market_price
                except (ValueError, IndexError):
                    continue
                    
    except FileNotFoundError:
        print(f"Warning: Could not find file {csv_filename}")
        return 1e6
    
    return total_error


def create_synthetic_price_matrix(theta: List[float], s0: float, r: float,
                                 k_values: List[float], t_values: List[float]) -> List[List[float]]:
    """
    Generate synthetic option price matrix using Heston model.
    
    This function creates perfect calibration data for testing purposes.
    
    Parameters
    ----------
    theta : list
        Heston parameters [kappa, v_bar, zeta, v0, rho]
    s0 : float
        Initial stock price
    r : float
        Risk-free rate
    k_values : list
        Array of strike prices
    t_values : list
        Array of time to maturity values
        
    Returns
    -------
    list of lists
        2D matrix of option prices P[i][j] for strike K[i] and maturity T[j]
    """
    kappa, v_bar, zeta, v0, rho = theta
    price_matrix = []
    
    for i, k in enumerate(k_values):
        price_row = []
        for t in t_values:
            price = european_call_price(s0, k, t, v0, v_bar, kappa, zeta, r, rho)
            price_row.append(price)
        price_matrix.append(price_row)
    
    return price_matrix


def iteration_callback(xk: np.ndarray, convergence: float) -> None:
    """
    Callback function called after each iteration of differential evolution.
    
    Parameters
    ----------
    xk : np.ndarray
        Current parameter vector
    convergence : float
        Current convergence value
    """
    print(f"Current parameters: {xk}")
    print(f"Convergence: {convergence}")
    print(f"Time elapsed: {time.time() - iteration_callback.start_time:.2f} seconds\n")


def read_price_data_from_csv(ticker: str) -> Tuple[List[float], List[List[float]]]:
    """
    Read option price data from CSV file.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol for the data file
        
    Returns
    -------
    tuple
        (strike_prices, price_matrix) where price_matrix[i][j] is the price
        for strike i and maturity j
        
    Raises
    ------
    FileNotFoundError
        If the specified file cannot be found
    """
    filename = f"data/{ticker}.csv"
    k_values = []
    price_matrix = []
    
    try:
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            
            for i, row in enumerate(reader):
                if i == 0:
                    # First row contains strike prices
                    k_values = [float(k) for k in row]
                else:
                    # Subsequent rows contain prices for each maturity
                    price_row = [float(p) for p in row]
                    price_matrix.append(price_row)
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}")
    except ValueError as e:
        raise ValueError(f"Error parsing CSV data: {e}")
    
    return k_values, price_matrix


def generate_random_heston_parameters(lower_bounds: List[float], upper_bounds: List[float],
                                     n_sets: int, seed: Optional[int] = None) -> List[List[float]]:
    """
    Generate random sets of Heston parameters within specified bounds.
    
    Parameters
    ----------
    lower_bounds : list
        Lower bounds for [kappa, v_bar, zeta, v0, rho]
    upper_bounds : list
        Upper bounds for [kappa, v_bar, zeta, v0, rho]
    n_sets : int
        Number of parameter sets to generate
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list of lists
        List of parameter sets, each containing [kappa, v_bar, zeta, v0, rho]
    """
    if seed is not None:
        np.random.seed(seed)
    
    parameter_sets = []
    
    for i in range(n_sets):
        parameter_set = []
        for j in range(len(lower_bounds)):
            lower = lower_bounds[j]
            upper = upper_bounds[j]
            random_value = np.random.random()
            parameter = round(lower + random_value * (upper - lower), 4)
            parameter_set.append(parameter)
        
        parameter_sets.append(parameter_set)
        print(f"Parameter set {i+1}: {parameter_set}")
    
    return parameter_sets


def calibrate_heston_parameters(s0: float, r: float, k_values: List[float],
                               t_values: List[float], price_matrix: List[List[float]],
                               parameter_bounds: Tuple[List[float], List[float]],
                               objective_function: str = 'squared_error',
                               max_iterations: int = 1000,
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Calibrate Heston model parameters using differential evolution.
    
    Parameters
    ----------
    s0 : float
        Initial stock price
    r : float
        Risk-free rate
    k_values : list
        Array of strike prices
    t_values : list
        Array of time to maturity values
    price_matrix : list of lists
        2D matrix of market option prices
    parameter_bounds : tuple
        (lower_bounds, upper_bounds) for [kappa, v_bar, zeta, v0, rho]
    objective_function : str, optional
        Type of objective function ('squared_error' or 'absolute_error')
    max_iterations : int, optional
        Maximum number of iterations
    verbose : bool, optional
        Whether to print progress information
        
    Returns
    -------
    dict
        Calibration results including optimal parameters and objective value
    """
    lower_bounds, upper_bounds = parameter_bounds
    bounds = Bounds(lower_bounds, upper_bounds)
    
    # Select objective function
    if objective_function == 'squared_error':
        obj_func = squared_error_objective
    elif objective_function == 'absolute_error':
        obj_func = absolute_error_objective
    else:
        raise ValueError("objective_function must be 'squared_error' or 'absolute_error'")
    
    # Set up callback if verbose
    callback = None
    if verbose:
        iteration_callback.start_time = time.time()
        callback = iteration_callback
    
    # Run differential evolution
    result = differential_evolution(
        obj_func,
        bounds,
        args=(s0, r, k_values, t_values, price_matrix),
        maxiter=max_iterations,
        disp=verbose,
        callback=callback,
        seed=42  # For reproducible results
    )
    
    return {
        'success': result.success,
        'optimal_parameters': result.x,
        'objective_value': result.fun,
        'iterations': result.nit,
        'message': result.message,
        'full_result': result
    }


def save_calibration_results(calibration_results: Dict[str, Any], s0: float, r: float,
                            market_data_file: str, output_file: str) -> None:
    """
    Save calibration results to CSV file with comparison to market prices.
    
    Parameters
    ----------
    calibration_results : dict
        Results from calibrate_heston_parameters
    s0 : float
        Initial stock price
    r : float
        Risk-free rate
    market_data_file : str
        Path to market data CSV file
    output_file : str
        Path to output CSV file
    """
    theta = calibration_results['optimal_parameters']
    kappa, v_bar, zeta, v0, rho = theta
    
    with open(output_file, 'w', newline='') as csvf:
        writer = csv.writer(csvf, delimiter=",")
        writer.writerow(["Strike", "Maturity", "Market Price", "Model Price", "Absolute Error"])
        
        try:
            with open(market_data_file, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=",")
                next(reader)  # Skip header row
                
                for row in reader:
                    try:
                        k = float(row[0])
                        t = float(row[1])
                        market_price = float(row[2])
                        
                        model_price = european_call_price(s0, k, t, v0, v_bar, kappa, zeta, r, rho)
                        abs_error = abs(model_price - market_price)
                        
                        writer.writerow([k, t, market_price, model_price, abs_error])
                    except (ValueError, IndexError):
                        continue
                        
        except FileNotFoundError:
            print(f"Warning: Could not find market data file {market_data_file}")


def main():
    """Main function for testing and demonstration."""
    # Example parameters
    s0 = 100.0
    r = 0.05
    
    # Parameter bounds: [kappa, v_bar, zeta, v0, rho]
    lower_bounds = [0.1, 0.01, 0.1, 0.01, -0.9]
    upper_bounds = [10.0, 0.5, 2.0, 0.5, 0.9]
    
    # Generate some test data
    true_params = [2.0, 0.04, 0.3, 0.04, -0.7]
    k_values = [90, 95, 100, 105, 110]
    t_values = [0.25, 0.5, 1.0]
    
    # Create synthetic market data
    price_matrix = create_synthetic_price_matrix(true_params, s0, r, k_values, t_values)
    
    print("Starting Heston parameter calibration...")
    print(f"True parameters: {true_params}")
    
    # Calibrate parameters
    results = calibrate_heston_parameters(
        s0, r, k_values, t_values, price_matrix,
        (lower_bounds, upper_bounds),
        max_iterations=100,
        verbose=True
    )
    
    print("\nCalibration Results:")
    print(f"Success: {results['success']}")
    print(f"Optimal parameters: {results['optimal_parameters']}")
    print(f"Objective value: {results['objective_value']}")


if __name__ == "__main__":
    main()


# Legacy function aliases for backward compatibility
def ObjFuncSqr(theta, S0, r, K, T, P):
    """Legacy alias for squared_error_objective."""
    return squared_error_objective(theta, S0, r, K, T, P)


def ObjFuncAbs(theta, S0, r, K, T, P):
    """Legacy alias for absolute_error_objective."""
    return absolute_error_objective(theta, S0, r, K, T, P)


def MktObjFuncSqr(theta, S0, r):
    """Legacy alias for market_data_squared_error_objective."""
    return market_data_squared_error_objective(theta, S0, r, "data/AppleYear.csv")


def MktObjFuncRelAbs(theta, S0, r):
    """Legacy alias for market_data_relative_absolute_error_objective."""
    return market_data_relative_absolute_error_objective(theta, S0, r, "data/Appeltje.csv")


def CreateP(theta, S0, r, K, T):
    """Legacy alias for create_synthetic_price_matrix."""
    return create_synthetic_price_matrix(theta, S0, r, K, T)


def Iteration(xk, convergence):
    """Legacy alias for iteration_callback."""
    iteration_callback(xk, convergence)


def ReadP(ticker):
    """Legacy alias for read_price_data_from_csv."""
    return read_price_data_from_csv(ticker)


def RandomHP(lb, ub, N):
    """Legacy alias for generate_random_heston_parameters."""
    # Fix the seeding issue from the original code
    return generate_random_heston_parameters(lb, ub, N, seed=42)
