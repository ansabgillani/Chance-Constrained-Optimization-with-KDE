"""Canonical two-state lunar landing benchmark used by the experiments.

The benchmark follows the model documented in ``Research_Report_2.md``:
``y_dot=v``, ``v_dot=-1.622+u``, nonnegative thrust bounded by three, a
terminal position event ``|y(tf)-xi_1| <= delta`` and a node-wise thrust
 event ``u(t)+xi_2 <= 3``. This module defines only the model and residuals;
sampling is performed by the experiment runner.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
import numpy as np


def _error_vector(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar, (n,), or (n, 1) array")
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be nonempty and finite")
    return arr


@dataclass
class LunarLandingProblem:
    """Finite-horizon lunar landing model with published benchmark parameters."""
    tf: float = 20.0
    u_max: float = 3.0
    delta: float = 0.25
    epsilon_a: float = 0.10
    epsilon_b: float = 0.01
    terminal_sigma: float = 0.10
    gravity: float = 1.622
    initial_state: tuple[float, float] = (100.0, 0.0)
    terminal_state: tuple[float, float] = (0.0, 0.0)
    metadata: Mapping[str, object] = field(default_factory=dict)
    n_states: int = 2
    n_controls: int = 1
    uncertainty_dim: int = 1
    n_terminal_constraints: int = 1
    n_path_constraints: int = 1

    def __post_init__(self) -> None:
        values = (self.tf, self.u_max, self.delta, self.epsilon_a,
                  self.epsilon_b, self.terminal_sigma, self.gravity)
        if any(not np.isfinite(float(v)) for v in values):
            raise ValueError("lunar parameters must be finite")
        if self.tf <= 0 or self.u_max <= 0 or self.delta <= 0 or self.gravity <= 0:
            raise ValueError("tf, u_max, delta, and gravity must be positive")
        if self.terminal_sigma < 0:
            raise ValueError("terminal_sigma must be nonnegative")
        if not (0 < self.epsilon_a < 1 and 0 < self.epsilon_b < 1):
            raise ValueError("epsilon_a and epsilon_b must lie in (0, 1)")
        if len(self.initial_state) != 2 or len(self.terminal_state) != 2:
            raise ValueError("initial_state and terminal_state must have length two")
        if not np.all(np.isfinite(self.initial_state)) or not np.all(np.isfinite(self.terminal_state)):
            raise ValueError("initial_state and terminal_state must be finite")
        self.metadata = {
            "name": "lunar_landing", "gravity": float(self.gravity),
            "tf": float(self.tf), "u_max": float(self.u_max),
            "delta": float(self.delta), "epsilon_a": float(self.epsilon_a),
            "epsilon_b": float(self.epsilon_b), "terminal_sigma": float(self.terminal_sigma),
            "terminal_error_distribution": "Normal(0, 0.1^2)",
            "path_error_distribution": "bimodal Gaussian mixture (experiment sampler)",
            "published_model_source": "Research_Report_2.md; Keil et al. benchmark summary",
            "residual_convention": "residual <= 0 is safe", "uncertainty_dim": 1,
            "n_terminal_constraints": 1, "n_path_constraints": 1,
        }

    @property
    def control_bounds(self) -> tuple[float, float]:
        return 0.0, float(self.u_max)

    def dynamics(self, t, state, control, disturbance=0.0) -> np.ndarray:
        del t
        state_arr = np.asarray(state, dtype=float)
        control_arr = np.asarray(control, dtype=float)
        disturbance_arr = np.asarray(disturbance, dtype=float)
        if state_arr.shape[-1:] != (2,) or not np.all(np.isfinite(state_arr)):
            raise ValueError("state must have a finite trailing dimension of size 2")
        if not np.all(np.isfinite(control_arr)) or not np.all(np.isfinite(disturbance_arr)):
            raise ValueError("control and disturbance must be finite")
        out = np.empty(np.broadcast(state_arr[..., 0], state_arr[..., 1], control_arr, disturbance_arr).shape + (2,))
        out[..., 0] = state_arr[..., 1]
        out[..., 1] = -self.gravity + control_arr + disturbance_arr
        return out

    def objective(self, states, controls, time=None) -> float:
        del states
        u = np.asarray(controls, dtype=float).reshape(-1)
        if u.size == 0 or not np.all(np.isfinite(u)):
            raise ValueError("controls must be nonempty and finite")
        if time is None:
            return float(np.mean(u))
        t = np.asarray(time, dtype=float).reshape(-1)
        if t.size not in (u.size, u.size + 1) or t.size < 2 or not np.all(np.isfinite(t)):
            raise ValueError("time must have one value per control node or interval")
        if np.any(np.diff(t) <= 0):
            raise ValueError("time must be strictly increasing")
        if u.size == t.size - 1:
            return float(np.sum(u * np.diff(t)))
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(trap(u, t))

    def deterministic_constraints(self, states, controls) -> np.ndarray:
        """Return inequalities in SciPy convention (nonnegative is feasible)."""
        X, U = np.asarray(states, dtype=float), np.asarray(controls, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] == 0:
            raise ValueError("states must have shape (n_nodes, 2)")
        return np.concatenate((X[0] - np.asarray(self.initial_state),
                               np.asarray(self.terminal_state) - X[-1], U, self.u_max - U))

    def terminal_residual(self, state, xi) -> np.ndarray:
        """Terminal residual ``abs(y(tf)-xi_1)-delta``; nonpositive is safe."""
        s = np.asarray(state, dtype=float)
        if s.shape != (2,) or not np.all(np.isfinite(s)):
            raise ValueError("state must be a finite vector with shape (2,)")
        z = _error_vector(xi, name="xi")
        return np.abs(s[0] - z) - self.delta

    def path_residual(self, control, xi) -> np.ndarray:
        """Path residual ``u+xi_2-u_max``; nonpositive is safe."""
        z = _error_vector(xi, name="xi")
        u = np.asarray(control, dtype=float)
        if not np.all(np.isfinite(u)):
            raise ValueError("control must be finite")
        if u.ndim == 0:
            values = float(u) + z - self.u_max
        elif u.ndim == 1 and u.size == z.size:
            values = u + z - self.u_max
        else:
            raise ValueError("control must be scalar or have one value per uncertainty sample")
        return np.asarray(values, dtype=float)

    def path_residual_nodes(self, controls, xi) -> np.ndarray:
        """Node-by-sample path residuals with shape ``(n_nodes, n_samples)``."""
        U = np.asarray(controls, dtype=float).reshape(-1)
        if U.size == 0 or not np.all(np.isfinite(U)):
            raise ValueError("controls must be nonempty and finite")
        z = _error_vector(xi, name="xi")
        return U[:, None] + z[None, :] - self.u_max

    def uncertain_residuals(self, state, xi) -> np.ndarray:
        return self.terminal_residual(state, xi)


def make_lunar_landing(**kwargs) -> LunarLandingProblem:
    return LunarLandingProblem(**kwargs)


LunarLanding = LunarLandingProblem
