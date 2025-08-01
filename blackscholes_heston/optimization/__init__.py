"""
Optimization and calibration module.

This module contains optimization algorithms and calibration methods
for financial model parameters.
"""

from .bisection import *
from .differential_evolution import *

__all__ = ["bisection", "differential_evolution"]
