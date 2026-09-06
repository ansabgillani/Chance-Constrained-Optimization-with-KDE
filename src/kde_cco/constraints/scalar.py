"""Scalar chance-constraint estimators.

The residual convention used throughout the package is ``r(x, xi) <= 0`` is
safe and ``r(x, xi) > 0`` is a violation.  For residual samples ``r_i``, the
integrated KDE estimate of the safe probability is therefore
``mean(K(-r_i / h))``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import numpy as np

from ..kde.kernels import (
    biased_epanechnikov_cdf, epanechnikov_cdf, gaussian_cdf, gaussian_pdf,
)


ArrayLike = np.ndarray | float | list[float] | tuple[float, ...]


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be real-valued") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def residuals(decision: ArrayLike, uncertainty: ArrayLike) -> np.ndarray:
    """Evaluate an additive residual elementwise with NumPy broadcasting."""
    x = _finite_array(decision, name="decision")
    xi = _finite_array(uncertainty, name="uncertainty")
    try:
        out = np.add(x, xi)
    except ValueError as exc:
        raise ValueError("decision and uncertainty shapes are not broadcastable") from exc
    return np.asarray(out, dtype=float)


def _validate_residuals(values: ArrayLike) -> tuple[np.ndarray, bool]:
    arr = _finite_array(values, name="residuals")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim not in (1, 2) or arr.shape[0] == 0:
        raise ValueError("residuals must be nonempty and one- or two-dimensional")
    return arr, arr.ndim == 1


def _reduce_sample_axis(values: np.ndarray, was_vector: bool) -> float | np.ndarray:
    out = np.mean(values, axis=0)
    return float(out) if was_vector else np.asarray(out, dtype=float)


def empirical_violation(residual_values: ArrayLike) -> float | np.ndarray:
    """Return the empirical probability of strict residual violation ``r > 0``."""
    r, was_vector = _validate_residuals(residual_values)
    return _reduce_sample_axis(r > 0.0, was_vector)


def _cdf(kernel: str | Callable[[np.ndarray], np.ndarray], z: np.ndarray) -> np.ndarray:
    if callable(kernel):
        out = np.asarray(kernel(z), dtype=float)
    elif kernel == "gaussian":
        out = gaussian_cdf(z)
    elif kernel == "epanechnikov":
        out = epanechnikov_cdf(z)
    else:
        raise ValueError("kernel must be 'gaussian', 'epanechnikov', or a callable")
    if out.shape != z.shape or not np.all(np.isfinite(out)):
        raise ValueError("kernel CDF must return finite values with matching shape")
    if np.any(out < -1e-12) or np.any(out > 1.0 + 1e-12):
        raise ValueError("kernel CDF must return values in [0, 1]")
    return np.clip(out, 0.0, 1.0)


def _validate_bandwidth(h: float) -> float:
    if not np.isfinite(h) or h <= 0:
        raise ValueError("bandwidth h must be positive and finite")
    return float(h)


def kde_safe_probability(residual_values: ArrayLike, h: float,
                         kernel: str | Callable[[np.ndarray], np.ndarray] = "gaussian") -> float | np.ndarray:
    """Estimate ``P(residual <= 0)`` with an integrated KDE."""
    r, was_vector = _validate_residuals(residual_values)
    h = _validate_bandwidth(h)
    return _reduce_sample_axis(_cdf(kernel, -r / h), was_vector)


def kde_violation_upper(residual_values: ArrayLike, h: float,
                        kernel: str | Callable[[np.ndarray], np.ndarray] = "epanechnikov",
                        bias: bool = False) -> float | np.ndarray:
    """Estimate violation probability, optionally using the conservative bias.

    ``bias=True`` uses the *local shifted Epanechnikov surrogate* implemented
    in this package.  Its safe contribution is zero for every positive
    residual, so ``1 - mean(K_B(-r_i/h))`` is pointwise no smaller than the
    empirical strict-violation probability.  This establishes dominance for
    this surrogate; it is not a claim that this translation reproduces Keil
    et al.'s Split-Bernstein construction, and the bound does not apply to an
    ordinary KDE.
    """
    r, was_vector = _validate_residuals(residual_values)
    h = _validate_bandwidth(h)
    if bias:
        if not isinstance(kernel, str) or kernel.lower() != "epanechnikov":
            raise ValueError("the conservative biased surrogate requires the Epanechnikov kernel")
        safe = np.mean(biased_epanechnikov_cdf(-r / h))
    else:
        safe = np.mean(_cdf(kernel, -r / h), axis=0)
    out = 1.0 - safe
    return float(out) if was_vector else np.asarray(out, dtype=float)


def kde_safe_probability_gradient(residual_values: ArrayLike,
                                  residual_jacobian: ArrayLike,
                                  h: float) -> float | np.ndarray:
    """Analytic chain derivative of Gaussian KDE safe probability.

    The Jacobian has shape ``(n_samples,)`` for one decision variable or
    ``(n_samples, n_variables)`` for several variables.  For
    ``p=N^-1 sum Phi(-r_i/h)``, ``dp/dx=-N^-1 sum phi(r_i/h) dr_i/dx/h``.
    """
    r, was_vector = _validate_residuals(residual_values)
    if not was_vector:
        raise ValueError("analytic gradient requires one-dimensional residual samples")
    jac = _finite_array(residual_jacobian, name="residual_jacobian")
    if jac.ndim == 0:
        jac = jac.reshape(1)
    if jac.ndim == 1:
        if jac.shape[0] != r.shape[0]:
            raise ValueError("residual_jacobian sample dimension does not match residuals")
        jac = jac[:, None]
        scalar_output = True
    elif jac.ndim == 2:
        if jac.shape[0] != r.shape[0] or jac.shape[1] == 0:
            raise ValueError("residual_jacobian must have shape (n_samples, n_variables)")
        scalar_output = False
    else:
        raise ValueError("residual_jacobian must be one- or two-dimensional")
    h = _validate_bandwidth(h)
    # gaussian_pdf(..., h) is the scaled density and already contains 1 / h.
    weights = gaussian_pdf(-r, h=h)[:, None]
    gradient = -np.mean(weights * jac, axis=0)
    return float(gradient[0]) if scalar_output else np.asarray(gradient, dtype=float)


gaussian_kde_safe_probability_gradient = kde_safe_probability_gradient
