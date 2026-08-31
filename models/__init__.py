"""
Package `models` : classes des estimateurs de matrice de covariance.
 
Exporte directement les éléments de base.py et tools.py.
"""

# from .base import CorrelationEstimator, NotFittedError
# from tools.tools import FloatArray, sample_correlation_matrix, center, normalise, frobenius_norm, frobenius_error
__all__ = [
    "CorrelationEstimator",
    "FloatArray",
    "NotFittedError",
    "center",
    "frobenius_error",
    "frobenius_norm",
    "normalise",
    "sample_correlation_matrix",
]