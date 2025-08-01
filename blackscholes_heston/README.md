# BlackScholes-Heston Financial Modeling Library

A comprehensive Python library for financial modeling using Black-Scholes and Heston stochastic volatility models, refactored to follow PEP 8 standards and modern Python best practices.

## Features

- **Black-Scholes Model**: European option pricing with surface plotting capabilities
- **Heston Stochastic Volatility Model**: Advanced option pricing with stochastic volatility
- **Monte Carlo Simulation**: Pricing of exotic options (Asian, Barrier, Lookback)
- **Parameter Calibration**: Differential evolution optimization for model parameters
- **Stochastic Processes**: Brownian motion generators and related processes
- **Implied Volatility**: Bisection method for implied volatility calculation

## Installation

The library is organized as a Python package. To use it, ensure you have the required dependencies:

```bash
pip install numpy scipy matplotlib
```

## Package Structure

```
blackscholes_heston/
├── __init__.py                 # Main package initialization
├── models/                     # Pricing models
│   ├── __init__.py
│   ├── black_scholes.py       # Black-Scholes model
│   └── heston.py              # Heston stochastic volatility model
├── stochastic/                # Stochastic processes
│   ├── __init__.py
│   └── brownian_motion.py     # Brownian motion generators
├── optimization/              # Optimization and calibration
│   ├── __init__.py
│   ├── bisection.py          # Implied volatility calculation
│   └── differential_evolution.py  # Parameter calibration
└── simulation/                # Monte Carlo simulation
    ├── __init__.py
    └── monte_carlo.py         # Exotic option pricing
```

## Quick Start

### Black-Scholes Option Pricing

```python
from blackscholes_heston.models.black_scholes import european_call_price, european_put_price

# Parameters
s0 = 100    # Current stock price
k = 100     # Strike price
t = 1       # Time to maturity (years)
r = 0.05    # Risk-free rate
sigma = 0.2 # Volatility

# Calculate option prices
call_price = european_call_price(s0, k, t, r, sigma)
put_price = european_put_price(s0, k, t, r, sigma)

print(f"Call price: ${call_price:.2f}")
print(f"Put price: ${put_price:.2f}")
```

### Heston Model Option Pricing

```python
from blackscholes_heston.models.heston import european_call_price, check_feller_condition

# Heston parameters
s0 = 100      # Current stock price
k = 100       # Strike price
t = 1         # Time to maturity
v0 = 0.04     # Initial variance
v_bar = 0.04  # Long-term variance
kappa = 2.0   # Mean reversion speed
zeta = 0.3    # Volatility of variance
r = 0.05      # Risk-free rate
rho = -0.7    # Correlation

# Check Feller condition
feller_ok = check_feller_condition(kappa, v_bar, zeta)
print(f"Feller condition satisfied: {feller_ok}")

# Calculate option price
heston_price = european_call_price(s0, k, t, v0, v_bar, kappa, zeta, r, rho)
print(f"Heston call price: ${heston_price:.2f}")
```

### Monte Carlo Simulation

```python
from blackscholes_heston.simulation.monte_carlo import (
    european_call_payoff, monte_carlo_option_pricing
)
import numpy as np

# Monte Carlo parameters
n_simulations = 10000
discount_factor = np.exp(-r * t)

# Run Monte Carlo simulation
results = monte_carlo_option_pricing(
    european_call_payoff, n_simulations, discount_factor,
    s0, k, t, v0, v_bar, kappa, zeta, r, rho, 1000
)

print(f"MC Price: ${results['price']:.2f}")
print(f"95% CI: [{results['confidence_interval_95'][0]:.2f}, {results['confidence_interval_95'][1]:.2f}]")
```

### Parameter Calibration

```python
from blackscholes_heston.optimization.differential_evolution import calibrate_heston_parameters

# Market data (strikes, maturities, prices)
k_values = [90, 95, 100, 105, 110]
t_values = [0.25, 0.5, 1.0]
price_matrix = [[...], [...], ...]  # Your market prices

# Parameter bounds [kappa, v_bar, zeta, v0, rho]
lower_bounds = [0.1, 0.01, 0.1, 0.01, -0.9]
upper_bounds = [10.0, 0.5, 2.0, 0.5, 0.9]

# Calibrate parameters
results = calibrate_heston_parameters(
    s0, r, k_values, t_values, price_matrix,
    (lower_bounds, upper_bounds)
)

print(f"Optimal parameters: {results['optimal_parameters']}")
```

## Backward Compatibility

The library maintains backward compatibility with the original codebase through legacy function aliases. You can still use the original function names:

```python
# Legacy usage (still works)
import Heston as h
import BlackScholes as bs

call_price = bs.EuroCall(100, 100, 1, 0.05, 0.2)
heston_price = h.EuroCall(100, 100, 1, 0.04, 0.04, 2.0, 0.3, 0.05, -0.7)
```

## Key Improvements

### PEP 8 Compliance
- **Function Names**: Changed from `CamelCase` to `snake_case` (e.g., `EuroCall` → `european_call_price`)
- **Variable Names**: Descriptive names following Python conventions
- **Documentation**: Comprehensive docstrings with parameter descriptions
- **Type Hints**: Added type annotations for better code clarity
- **Line Length**: All lines under 88 characters
- **Import Organization**: Properly organized imports

### Code Organization
- **Modular Structure**: Separated concerns into logical modules
- **Package Structure**: Proper Python package with `__init__.py` files
- **Error Handling**: Improved error handling and validation
- **Code Reuse**: Eliminated code duplication

### Enhanced Functionality
- **Better Documentation**: Comprehensive docstrings and examples
- **Improved Error Messages**: More descriptive error messages
- **Validation**: Input parameter validation
- **Flexibility**: More configurable functions with optional parameters

## Testing

Run the regression tests to verify functionality:

```bash
python test_regression.py
```

All tests should pass, confirming that the refactored code produces identical results to the original implementation.

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Scientific computing (optimization, integration, statistics)
- `matplotlib`: Plotting and visualization

## License

This project maintains the same license as the original codebase.

## Contact

Email: sanderkorteweg@gmail.com

## Migration Guide

If you're migrating from the old codebase:

1. **Import Changes**: Update imports to use the new package structure
2. **Function Names**: Use new snake_case function names or keep legacy aliases
3. **Parameters**: Some functions now have more descriptive parameter names
4. **Error Handling**: Be prepared for more descriptive error messages

The legacy aliases ensure your existing code will continue to work without modifications.
