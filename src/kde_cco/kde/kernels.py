"""One-dimensional kernels and their integrated kernels.

The functions use the chance-constraint convention: ``u`` is a standardized
residual and an optional ``h`` evaluates the scaled kernel
``k_h(x) = k(x/h)/h`` and CDF ``K_h(x) = K(x/h)``.  A safe contribution for
residual ``r`` is consequently ``K(-r/h)``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def _finite_array(value: np.ndarray | float, *, name: str = "value") -> np.ndarray:
    """Convert a scalar or array to finite floating point values."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be real-valued") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _bandwidth(h: float) -> float:
    try:
        value = float(h)
    except (TypeError, ValueError) as exc:
        raise ValueError("bandwidth h must be positive and finite") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("bandwidth h must be positive and finite")
    return value


def gaussian_pdf(u: np.ndarray | float, h: float | None = None) -> np.ndarray:
    """Evaluate the standard Gaussian density, optionally at bandwidth ``h``."""
    x = _finite_array(u, name="u")
    if h is not None:
        scale = _bandwidth(h)
        x = x / scale
        return np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi) * scale)
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def epanechnikov_pdf(u: np.ndarray | float, h: float | None = None) -> np.ndarray:
    """Evaluate the Epanechnikov density (support ``[-h, h]``)."""
    x = _finite_array(u, name="u")
    scale = 1.0 if h is None else _bandwidth(h)
    z = x / scale
    return 0.75 * np.maximum(0.0, 1.0 - z * z) * (np.abs(z) <= 1.0) / scale


def gaussian_cdf(u: np.ndarray | float, h: float | None = None) -> np.ndarray:
    """Evaluate the integrated Gaussian kernel ``K_h(u)``."""
    x = _finite_array(u, name="u")
    if h is not None:
        x = x / _bandwidth(h)
    return ndtr(x)


def epanechnikov_cdf(u: np.ndarray | float, h: float | None = None) -> np.ndarray:
    """Evaluate the integrated Epanechnikov kernel ``K_h(u)``."""
    x = _finite_array(u, name="u")
    if h is not None:
        x = x / _bandwidth(h)
    out = np.empty_like(x, dtype=float)
    low = x <= -1.0
    high = x >= 1.0
    middle = ~(low | high)
    out[low] = 0.0
    out[high] = 1.0
    z = x[middle]
    out[middle] = 0.5 + 0.75 * (z - z**3 / 3.0)
    return out


def biased_epanechnikov_cdf(u: np.ndarray | float, h: float | None = None) -> np.ndarray:
    """Evaluate the one-sided conservative Epanechnikov integrated kernel.

    The translate has support ``[0, 2]`` in standardized coordinates.  Hence
    ``K_B(-r/h) <= 1{r <= 0}`` pointwise, so one minus its sample mean is an
    upper surrogate for the violation probability.
    """
    x = _finite_array(u, name="u")
    if h is not None:
        x = x / _bandwidth(h)
    return epanechnikov_cdf(x - 1.0)
