"""
BlackScholes-Heston Financial Modeling Library

A comprehensive Python library for financial modeling using Black-Scholes
and Heston stochastic volatility models.

Modules:
    models: Core pricing models (Black-Scholes, Heston)
    stochastic: Stochastic process generators
    optimization: Parameter calibration and optimization
    simulation: Monte Carlo simulation methods
"""

__version__ = "1.0.0"
__author__ = "Sander Korteweg"
__email__ = "sanderkorteweg@gmail.com"

# Import main components for easy access
from .models import black_scholes, heston
from .stochastic import brownian_motion
from .optimization import bisection, differential_evolution
from .simulation import monte_carlo

__all__ = [
    "black_scholes",
    "heston", 
    "brownian_motion",
    "bisection",
    "differential_evolution",
    "monte_carlo"
]
