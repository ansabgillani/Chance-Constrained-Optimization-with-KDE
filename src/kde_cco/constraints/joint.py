"""Joint chance-constraint aggregation functions."""

from __future__ import annotations

import numpy as np
from numpy.exceptions import AxisError
from scipy.special import logsumexp


def boole_risk_allocation(epsilon: float, count: int) -> np.ndarray:
    """Allocate total violation budget equally across ``count`` constraints."""
    if not np.isfinite(epsilon) or epsilon <= 0 or epsilon > 1:
        raise ValueError("epsilon must lie in (0, 1]")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
        raise ValueError("count must be a positive integer")
    return np.full(int(count), float(epsilon) / int(count), dtype=float)


def _matrix(residual_matrix: np.ndarray) -> np.ndarray:
    a = np.asarray(residual_matrix, dtype=float)
    if a.ndim not in (1, 2) or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("residual_matrix must be a nonempty one- or two-dimensional array")
    return a


def max_violation(residual_matrix: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Return the exact maximum residual, optionally along an axis."""
    try:
        return np.max(_matrix(residual_matrix), axis=axis)
    except (TypeError, ValueError, AxisError) as exc:
        raise ValueError("axis is invalid for residual_matrix") from exc


def smooth_max(values: np.ndarray, tau: float, axis: int | None = None) -> np.ndarray | float:
    """Compute ``tau * log(sum(exp(values / tau)))`` stably."""
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")
    a = _matrix(values)
    try:
        out = tau * logsumexp(a / tau, axis=axis)
    except (TypeError, ValueError, AxisError) as exc:
        raise ValueError("axis is invalid for residual_matrix") from exc
    # The subtraction-free expression is intentionally used instead of a
    # clipped approximation: for finite values log-sum-exp is a smooth upper
    # envelope of max and is useful as a differentiable joint surrogate.
    return out
