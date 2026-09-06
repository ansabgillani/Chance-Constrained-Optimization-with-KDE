"""Scalar and joint chance-constraint utilities."""

from .scalar import (
    empirical_violation,
    gaussian_kde_safe_probability_gradient,
    kde_safe_probability,
    kde_safe_probability_gradient,
    kde_violation_upper,
    residuals,
)
from .joint import boole_risk_allocation, max_violation, smooth_max

__all__ = [
    "residuals", "empirical_violation", "kde_safe_probability", "kde_violation_upper",
    "kde_safe_probability_gradient", "gaussian_kde_safe_probability_gradient",
    "boole_risk_allocation", "max_violation", "smooth_max",
]
