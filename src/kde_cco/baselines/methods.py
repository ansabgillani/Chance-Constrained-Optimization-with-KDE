"""Deterministic baseline chance-constraint builders."""
from __future__ import annotations
import numpy as np


def _samples(residual_fn, x, samples):
    vals = np.asarray(residual_fn(np.asarray(x), np.asarray(samples)))
    if vals.ndim == 1:
        return vals
    return np.max(vals, axis=tuple(range(1, vals.ndim)))


def _result(method, constraint, **metadata):
    return {"method": method, "constraint": constraint, "constraints": constraint, "metadata": metadata}


def nominal(residual_fn, samples):
    """Mean-value (nominal) residual constraint builder."""
    mean = np.mean(np.asarray(samples), axis=0)
    return _result("nominal", lambda x: float(np.max(np.asarray(residual_fn(x, mean)))),
                   nominal_sample=mean)


def worst_case(residual_fn, samples):
    """Robust baseline: enforce every supplied scenario."""
    data = np.asarray(samples)
    return _result("worst_case", lambda x: float(np.max(np.asarray(residual_fn(x, data)))))


def scenario(residual_fn, samples):
    """Scenario baseline; equivalent to worst case over the training scenarios."""
    out = worst_case(residual_fn, samples)
    out["method"] = "scenario"
    return out


def empirical_quantile(residual_fn, samples, epsilon=0.05):
    """Empirical (1-epsilon) residual quantile; residual <= 0 is safe."""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0,1)")
    data = np.asarray(samples)
    return _result("empirical_quantile",
                   lambda x: float(np.quantile(_samples(residual_fn, x, data), 1.0 - epsilon)),
                   epsilon=float(epsilon))


def gaussian_parametric(residual_fn, samples, epsilon=0.05):
    """Gaussian parametric baseline for affine scalar residuals.

    Nonlinear or vector residuals are rejected explicitly because a scalar
    normal quantile is not a valid reformulation for those cases.
    """
    data = np.asarray(samples, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2 or data.shape[1] != 1:
        raise NotImplementedError("Gaussian reformulation supports only scalar affine residuals")
    if data.shape[0] < 2 or not np.all(np.isfinite(data)):
        raise ValueError("samples must contain at least two finite values")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0,1)")
    # A Gaussian quantile is valid here only when the residual is affine in the
    # scalar uncertainty.  Check that property at a fixed decision before
    # constructing the baseline; nonlinear residuals (including the canonical
    # static benchmark) must use KDE or scenarios instead.
    dim = data.shape[1]
    x_probe = np.zeros(2 if dim > 1 else 1, dtype=float)
    points = np.array([[0.0], [0.37], [-0.61], [0.37 - 0.61]], dtype=float)
    try:
        values = np.asarray(residual_fn(x_probe, points), dtype=float).reshape(-1)
    except Exception as exc:
        raise NotImplementedError(
            "Gaussian reformulation requires a scalar affine residual in uncertainty"
        ) from exc
    if values.size != points.shape[0] or not np.all(np.isfinite(values)):
        raise NotImplementedError(
            "Gaussian reformulation requires a scalar affine residual in uncertainty"
        )
    # f(a+b)-f(a)-f(b)+f(0) must vanish for an affine map.  Scale the check so
    # it remains stable for residuals with large offsets.
    affine_error = values[3] - values[1] - values[2] + values[0]
    scale = max(1.0, float(np.max(np.abs(values))))
    if abs(float(affine_error)) > 1e-8 * scale:
        raise NotImplementedError(
            "Gaussian reformulation requires a scalar affine residual in uncertainty"
        )
    from scipy.stats import norm
    mu, sd = float(np.mean(data[:, 0])), float(np.std(data[:, 0], ddof=1))
    z = float(norm.ppf(1.0 - epsilon))
    # The residual function is still evaluated at the mean; this is a safe
    # adapter for affine residuals and avoids silently pretending nonlinearity.
    def constraint(x):
        val = np.asarray(residual_fn(x, np.array([[mu]]))).reshape(-1)
        return float(val[0] + z * sd)
    return _result("gaussian_parametric", constraint, epsilon=float(epsilon), mean=mu, std=sd,
                   uncertainty_model="scalar_affine_gaussian")


def build_baseline(method, residual_fn, samples, epsilon=0.05):
    methods = {"nominal": nominal, "worst_case": worst_case, "scenario": scenario,
               "gaussian": gaussian_parametric, "gaussian_parametric": gaussian_parametric,
               "empirical": empirical_quantile, "empirical_quantile": empirical_quantile}
    try:
        fn = methods[method.lower()]
    except KeyError:
        raise ValueError(f"unknown baseline method: {method}") from None
    return fn(residual_fn, samples, epsilon) if method.lower() in {"gaussian", "gaussian_parametric", "empirical", "empirical_quantile"} else fn(residual_fn, samples)

# Explicit aliases used in experiment scripts.
nominal_constraints = nominal
worst_case_constraints = worst_case
scenario_constraints = scenario
gaussian_constraints = gaussian_parametric
empirical_quantile_constraints = empirical_quantile
