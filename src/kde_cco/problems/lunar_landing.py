"""Soft lunar-landing optimal-control benchmark.

States are altitude ``y`` and vertical velocity ``v``; control ``u`` is thrust.
The normalized dynamics are y_dot=v and v_dot=-1.622+u.  Disturbances are
represented in terminal altitude and path thrust residuals, with residual <= 0
meaning safe.
"""
from dataclasses import dataclass, field
from typing import Mapping
import numpy as np

@dataclass
class LunarLandingProblem:
    # A 20-unit horizon makes the stated initial/terminal conditions feasible
    # under nonnegative thrust and gravity 1.622.
    tf: float = 20.0
    u_max: float = 3.0
    delta: float = 0.25
    epsilon_a: float = 0.1
    epsilon_b: float = 0.01
    terminal_sigma: float = 0.1
    gravity: float = 1.622
    initial_state: tuple = (100.0, 0.0)
    terminal_state: tuple = (0.0, 0.0)
    bounds: tuple = ((0.0, None), (0.0, None))
    metadata: Mapping[str, object] = field(default_factory=dict)

    n_states: int = 2
    n_controls: int = 1

    def __post_init__(self):
        if self.tf <= 0 or self.u_max <= 0 or self.delta <= 0 or self.terminal_sigma < 0:
            raise ValueError("tf, u_max, delta, and terminal_sigma must be positive/nonnegative")
        if not (0 < self.epsilon_a < 1 and 0 < self.epsilon_b < 1):
            raise ValueError("epsilon_a and epsilon_b must lie in (0,1)")
        self.metadata = {
            "name": "lunar_landing", "gravity": self.gravity, "tf": self.tf,
            "u_max": self.u_max, "delta": self.delta,
            "epsilon_a": self.epsilon_a, "epsilon_b": self.epsilon_b,
            "terminal_sigma": self.terminal_sigma, "uncertainty_dim": 1,
            "n_terminal_constraints": 1, "n_path_constraints": 1,
            "residual_convention": "residual <= 0 is safe",
            "terminal_disturbance": "Gaussian",
            "path_disturbance": "bimodal renewable-style mixture",
        }

    uncertainty_dim: int = 1
    n_terminal_constraints: int = 1
    n_path_constraints: int = 1

    def dynamics(self, t, state, control, disturbance=0.0):
        y = np.asarray(state, dtype=float)
        u = np.asarray(control, dtype=float)
        out = np.empty(np.broadcast(y[..., 0], y[..., 1], u).shape + (2,))
        out[..., 0] = y[..., 1]
        out[..., 1] = -self.gravity + u + disturbance
        return out

    def objective(self, states, controls, time=None):
        u = np.asarray(controls, dtype=float)
        if time is None:
            return float(np.mean(u))
        return float(np.trapezoid(u.reshape(-1), np.asarray(time).reshape(-1)))

    def deterministic_constraints(self, states, controls):
        X, U = np.asarray(states), np.asarray(controls)
        return np.concatenate((X[0] - np.asarray(self.initial_state),
                               np.asarray(self.terminal_state) - X[-1],
                               U.reshape(-1), self.u_max - U.reshape(-1)))

    @property
    def control_bounds(self):
        return (0.0, self.u_max)

    def uncertain_residuals(self, state, xi):
        """Default uncertain residual callable (terminal event)."""
        return self.terminal_residual(state, xi)

    def terminal_residual(self, state, xi):
        """Published terminal event ``abs(y(tf) - xi) - delta``."""
        s, z = np.asarray(state, dtype=float), np.asarray(xi, dtype=float)
        if s.shape != (2,) or not np.all(np.isfinite(s)):
            raise ValueError("state must be a finite vector with shape (2,)")
        if z.ndim == 0:
            z = z.reshape(1)
        elif z.ndim == 2 and z.shape[1] == 1:
            z = z[:, 0]
        elif z.ndim != 1:
            raise ValueError("xi must be a scalar, (n,), or (n, 1) terminal-error array")
        if z.size == 0 or not np.all(np.isfinite(z)):
            raise ValueError("xi must be nonempty and finite")
        return np.abs(s[0] - z) - self.delta

    def path_residual(self, control, xi):
        u, z = np.asarray(control, dtype=float), np.asarray(xi, dtype=float)
        if z.ndim == 0:
            z = z.reshape(1)
        elif z.ndim == 2 and z.shape[1] == 1:
            z = z[:, 0]
        elif z.ndim != 1:
            raise ValueError("xi must be a scalar, (n,), or (n, 1) path-error array")
        if z.size == 0 or not np.all(np.isfinite(z)):
            raise ValueError("xi must be nonempty and finite")
        if u.ndim == 0:
            control = float(u)
        elif u.ndim == 1 and u.size == z.size:
            control = u
        else:
            raise ValueError("control must be scalar or have one value per uncertainty sample")
        return control + z - self.u_max


def make_lunar_landing(**kwargs):
    return LunarLandingProblem(**kwargs)

LunarLanding = LunarLandingProblem
