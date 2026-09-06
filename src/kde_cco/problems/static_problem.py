"""Canonical low-dimensional nonlinear chance-constrained benchmark."""
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional
import numpy as np

Array = np.ndarray

@dataclass
class StaticProblem:
    """Two-variable nonlinear benchmark.

    ``uncertain_residuals(x, xi)`` uses the convention residual <= 0 is safe;
    deterministic constraints use scipy's convention value >= 0 is feasible.
    """
    objective: Callable[[Array], float]
    deterministic_constraints: Callable[[Array], Array]
    uncertain_residuals: Callable[[Array, Array], Array]
    bounds: tuple
    metadata: Mapping[str, object] = field(default_factory=dict)
    n_variables: int = 2
    uncertainty_dim: int = 2
    n_uncertain_constraints: int = 1

    def residuals(self, x, xi):
        return self.uncertain_residuals(np.asarray(x), np.asarray(xi))


def make_static_problem(noise_scale: float = 0.25, radius: float = 1.0) -> StaticProblem:
    """Return a transparent synthetic nonlinear problem.

    The objective has a unique unconstrained minimizer near ``(0.8, 0.6)``.
    Uncertainty enters a circular safety residual, making the example useful
    for KDE, scenario, and parametric comparisons without external data.
    """
    sigma = float(noise_scale)
    radius = float(radius)
    if not np.isfinite(sigma) or sigma < 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("noise_scale must be finite and nonnegative; radius must be finite and positive")

    def objective(x):
        x = np.asarray(x, dtype=float)
        if x.shape != (2,) or not np.all(np.isfinite(x)):
            raise ValueError("decision must be a finite vector with shape (2,)")
        return float((x[0] - 0.8) ** 2 + (x[1] - 0.6) ** 2 + 0.05 * x[0] * x[1])

    def deterministic_constraints(x):
        x = np.asarray(x, dtype=float)
        if x.shape != (2,) or not np.all(np.isfinite(x)):
            raise ValueError("decision must be a finite vector with shape (2,)")
        # Keep decisions in a compact, nonnegative design region.
        return np.array([x[0], x[1], 2.0 - x[0], 2.0 - x[1], 1.8 - x[0] - x[1]])

    def uncertain_residuals(x, xi):
        x = np.asarray(x, dtype=float)
        z = np.asarray(xi, dtype=float)
        if x.shape != (2,) or not np.all(np.isfinite(x)):
            raise ValueError("decision must be a finite vector with shape (2,)")
        if z.ndim == 1:
            z = z.reshape(1, -1)
        if z.ndim != 2 or z.shape[1] != 2 or z.shape[0] == 0 or not np.all(np.isfinite(z)):
            raise ValueError("xi must be a nonempty finite array with shape (n, 2)")
        # Safety event: residual <= 0.  Broadcast over samples.
        return (x[0] + sigma * z[:, 0]) ** 2 + (x[1] + sigma * z[:, 1]) ** 2 - radius**2

    return StaticProblem(
        objective, deterministic_constraints, uncertain_residuals,
        bounds=((0.0, 2.0), (0.0, 2.0)),
        metadata={
            "name": "static_nonlinear", "noise_scale": sigma, "radius": radius,
            "n_variables": 2, "uncertainty_dim": 2, "n_uncertain_constraints": 1,
            "residual_convention": "residual <= 0 is safe",
        },
    )

# Backwards-compatible descriptive alias.
StaticNonlinearProblem = StaticProblem
