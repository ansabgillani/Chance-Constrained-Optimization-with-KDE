"""SciPy adapters for finite-dimensional chance-constrained benchmarks.

The problem objects deliberately expose solver-independent callables.  This
module translates their residual convention (``r <= 0`` is safe) to SciPy's
inequality convention (``g >= 0`` is feasible), and keeps the result schema
identical for the static and dispatch benchmarks.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any, TypedDict

import numpy as np


class SolverResult(TypedDict, total=False):
    """Stable serialisable subset of a SciPy optimisation result."""

    success: bool
    decision: np.ndarray
    objective: float
    nit: int
    message: str
    runtime_seconds: float


def _import_minimize():
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("scipy is required for the static and dispatch adapters") from exc
    return minimize


def _dimension(problem: Any) -> int:
    bounds = getattr(problem, "bounds", None)
    if bounds is None:
        raise TypeError("problem must expose finite-dimensional bounds")
    n = int(getattr(problem, "n_variables", len(bounds)))
    if n <= 0 or len(bounds) != n:
        raise ValueError("problem.n_variables must match the number of bounds")
    return n


def _bounds(problem: Any) -> tuple[tuple[float | None, float | None], ...]:
    raw = getattr(problem, "bounds", None)
    if raw is None:
        raise TypeError("problem must expose bounds")
    out: list[tuple[float | None, float | None]] = []
    for bound in raw:
        if bound is None or len(bound) != 2:
            raise ValueError("each bound must be a (lower, upper) pair")
        lo, hi = bound
        lo = None if lo is None else float(lo)
        hi = None if hi is None else float(hi)
        if lo is not None and not np.isfinite(lo):
            raise ValueError("lower bounds must be finite or None")
        if hi is not None and not np.isfinite(hi):
            raise ValueError("upper bounds must be finite or None")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("lower bound cannot exceed upper bound")
        out.append((lo, hi))
    return tuple(out)


def _initial_guess(problem: Any, initial_guess: Sequence[float] | None, bounds: tuple) -> np.ndarray:
    n = len(bounds)
    if initial_guess is None:
        values = []
        for lo, hi in bounds:
            if lo is not None and hi is not None:
                values.append((lo + hi) / 2.0)
            elif lo is not None:
                values.append(lo + 1.0)
            elif hi is not None:
                values.append(hi - 1.0)
            else:
                values.append(0.0)
        x0 = np.asarray(values, dtype=float)
    else:
        x0 = np.asarray(initial_guess, dtype=float)
        if x0.shape != (n,):
            raise ValueError(f"initial_guess must have shape ({n},)")
        x0 = x0.copy()
    if not np.all(np.isfinite(x0)):
        raise ValueError("initial_guess must be finite")
    # SLSQP rejects an initial point outside the explicit box.  Clipping is
    # deterministic and preserves the caller's direction while making the
    # adapter usable with common zero initialisations.
    lower = np.array([(-np.inf if lo is None else lo) for lo, _ in bounds])
    upper = np.array([(np.inf if hi is None else hi) for _, hi in bounds])
    return np.clip(x0, lower, upper)


def _vector_constraint(function: Callable[[np.ndarray], Any], *, name: str) -> Callable[[np.ndarray], np.ndarray]:
    def wrapped(x: np.ndarray) -> np.ndarray:
        values = np.asarray(function(np.asarray(x, dtype=float)), dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must return a finite scalar or one-dimensional array")
        return values
    return wrapped


def _constraint_from_chance(chance_constraint: Any) -> Callable[[np.ndarray], np.ndarray]:
    candidate = chance_constraint
    if isinstance(candidate, Mapping):
        for key in ("constraint", "constraints", "function", "fun"):
            if key in candidate:
                candidate = candidate[key]
                break
        else:
            raise TypeError("chance_constraint mapping must contain a callable constraint")
    elif hasattr(candidate, "constraint"):
        candidate = candidate.constraint
    if not callable(candidate):
        raise TypeError("chance_constraint must be callable or contain a callable 'constraint'")

    residual = _vector_constraint(candidate, name="chance_constraint")

    def feasible(x: np.ndarray) -> np.ndarray:
        # The package convention is residual <= 0.  Flip sign for SciPy.
        return -residual(x)

    return feasible


def _solve(problem: Any, *, initial_guess: Sequence[float] | None,
           chance_constraint: Any, method: str, tolerance: float,
           maxiter: int, options: Mapping[str, Any] | None) -> SolverResult:
    minimize = _import_minimize()
    n = _dimension(problem)
    bounds = _bounds(problem)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if int(maxiter) <= 0:
        raise ValueError("maxiter must be positive")
    x0 = _initial_guess(problem, initial_guess, bounds)

    objective = getattr(problem, "objective", None)
    if not callable(objective):
        raise TypeError("problem.objective must be callable")

    def objective_scalar(x: np.ndarray) -> float:
        value = np.asarray(objective(np.asarray(x, dtype=float)), dtype=float)
        if value.ndim != 0 or not np.isfinite(value):
            raise ValueError("problem.objective must return one finite scalar")
        return float(value)

    constraints: list[dict[str, Any]] = []
    deterministic = getattr(problem, "deterministic_constraints", None)
    if deterministic is not None:
        constraints.append({"type": "ineq", "fun": _vector_constraint(deterministic, name="deterministic_constraints")})
    if chance_constraint is not None:
        constraints.append({"type": "ineq", "fun": _constraint_from_chance(chance_constraint)})

    scipy_options = {"ftol": float(tolerance), "maxiter": int(maxiter)}
    if options:
        scipy_options.update(dict(options))
    started = perf_counter()
    result = minimize(objective_scalar, x0, method=method, bounds=bounds,
                      constraints=constraints, options=scipy_options)
    elapsed = perf_counter() - started
    return {
        "success": bool(result.success),
        "decision": np.asarray(result.x, dtype=float),
        "objective": float(result.fun),
        "nit": int(getattr(result, "nit", -1)),
        "message": str(result.message),
        "runtime_seconds": float(elapsed),
    }


def solve_static(problem: Any, initial_guess: Sequence[float] | None = None,
                 chance_constraint: Any = None, method: str = "SLSQP",
                 tolerance: float = 1e-8, maxiter: int = 500,
                 options: Mapping[str, Any] | None = None) -> SolverResult:
    """Solve a static nonlinear model with explicit bounds and constraints."""
    return _solve(problem, initial_guess=initial_guess,
                  chance_constraint=chance_constraint, method=method,
                  tolerance=tolerance, maxiter=maxiter, options=options)


def solve_dispatch(problem: Any, initial_guess: Sequence[float] | None = None,
                   chance_constraint: Any = None, method: str = "SLSQP",
                   tolerance: float = 1e-8, maxiter: int = 500,
                   options: Mapping[str, Any] | None = None) -> SolverResult:
    """Solve a two-generator dispatch model using the same adapter contract.

    ``EnergyDispatchProblem`` exposes the balance, nonnegativity and capacity
    inequalities through ``deterministic_constraints``.  Consequently this
    function runs an actual constrained optimisation and does not substitute a
    hand-picked dispatch vector.
    """
    return _solve(problem, initial_guess=initial_guess,
                  chance_constraint=chance_constraint, method=method,
                  tolerance=tolerance, maxiter=maxiter, options=options)


solve_static_problem = solve_static
solve = solve_static
