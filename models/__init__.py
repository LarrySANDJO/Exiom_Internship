"""
Package `models` : classes des estimateurs de matrice de covariance.

Exporte directement les éléments de base.py et tools.py.
"""

from tools.tools import (
    FloatArray,
    center,
    frobenius_error,
    frobenius_norm,
    normalise,
    sample_correlation_matrix,
)

from .base import CorrelationEstimator, NotFittedError

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