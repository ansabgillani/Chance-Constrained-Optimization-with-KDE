"""Synthetic two-generator dispatch benchmark with renewable uncertainty."""
from dataclasses import dataclass, field
from typing import Mapping
import numpy as np

@dataclass
class EnergyDispatchProblem:
    demand: float
    renewable_nominal: float
    capacities: tuple
    objective: callable
    deterministic_constraints: callable
    uncertain_residuals: callable
    bounds: tuple
    metadata: Mapping[str, object] = field(default_factory=dict)

    n_variables: int = 2
    uncertainty_dim: int = 1
    n_uncertain_constraints: int = 3

    def balance_residual(self, dispatch, renewable_error=0.0):
        x = np.asarray(dispatch, dtype=float)
        e = np.asarray(renewable_error, dtype=float)
        if x.shape != (2,) or not np.all(np.isfinite(x)):
            raise ValueError("dispatch must be a finite vector with shape (2,)")
        if not np.all(np.isfinite(e)):
            raise ValueError("renewable_error must be finite")
        return self.demand - self.renewable_nominal - e - np.sum(x, axis=-1)

    def capacity_residual(self, dispatch):
        x = np.asarray(dispatch, dtype=float)
        if x.shape != (2,) or not np.all(np.isfinite(x)):
            raise ValueError("dispatch must be a finite vector with shape (2,)")
        return x - np.asarray(self.capacities)

    def residuals(self, dispatch, xi):
        return self.uncertain_residuals(dispatch, xi)


def make_energy_dispatch(demand: float = 100.0, renewable_nominal: float = 30.0,
                         capacities=(90.0, 100.0), reserve: float = 0.0) -> EnergyDispatchProblem:
    """Create a two-generator dispatch model.

    Generator 1 is cheap but limited; generator 2 is more expensive.  Renewable
    forecast error is supplied as ``xi`` (positive means extra generation).
    Residuals are ``<= 0`` when balance and capacity are safe.
    """
    d, r = float(demand), float(renewable_nominal)
    caps = tuple(float(c) for c in capacities)
    if len(caps) != 2 or any(c <= 0 for c in caps):
        raise ValueError("capacities must contain two positive values")

    def objective(x):
        x = np.asarray(x, dtype=float)
        return float(0.018 * x[0] ** 2 + 12.0 * x[0] + 0.028 * x[1] ** 2 + 16.0 * x[1])

    def deterministic_constraints(x):
        x = np.asarray(x, dtype=float)
        # Generation nonnegative, capacity limits, and optional spinning reserve.
        return np.array([x[0], x[1], caps[0] - x[0], caps[1] - x[1],
                         x.sum() - (d - r + reserve)])

    def uncertain_residuals(x, xi):
        x = np.asarray(x, dtype=float)
        z = np.asarray(xi, dtype=float)
        if x.shape != (2,) or not np.all(np.isfinite(x)):
            raise ValueError("dispatch must be a finite vector with shape (2,)")
        if z.ndim == 0:
            z = z.reshape(1)
        elif z.ndim == 2 and z.shape[1] == 1:
            z = z[:, 0]
        elif z.ndim != 1:
            raise ValueError("xi must be a scalar, (n,), or (n, 1) renewable-error array")
        if z.size == 0 or not np.all(np.isfinite(z)):
            raise ValueError("xi must be nonempty and finite")
        balance = d - r - z - np.sum(x)
        # Include capacity residuals to support joint chance constraints.
        return np.column_stack((np.broadcast_to(balance, z.shape),
                                np.broadcast_to(np.asarray(x)[0] - caps[0], z.shape),
                                np.broadcast_to(np.asarray(x)[1] - caps[1], z.shape)))

    if not np.isfinite(d) or not np.isfinite(r) or d < 0 or r < 0:
        raise ValueError("demand and renewable_nominal must be finite and nonnegative")
    reserve = float(reserve)
    if not np.isfinite(reserve) or reserve < 0:
        raise ValueError("reserve must be finite and nonnegative")
    return EnergyDispatchProblem(
        d, r, caps, objective, deterministic_constraints, uncertain_residuals,
        ((0.0, caps[0]), (0.0, caps[1])),
        {
            "name": "energy_dispatch", "demand": d,
            "renewable_nominal": r, "capacities": caps, "reserve": reserve,
            "n_variables": 2, "uncertainty_dim": 1, "n_uncertain_constraints": 3,
            "residual_convention": "residual <= 0 is safe",
            "components": ("balance_deficit", "generator_1_capacity", "generator_2_capacity"),
        },
    )

EnergyDispatch = EnergyDispatchProblem
