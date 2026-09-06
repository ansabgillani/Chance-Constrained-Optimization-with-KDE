"""Local direct-collocation adapter for the lunar benchmark.

The transcription is intentionally small and inspectable. It uses forward
Euler defects and SciPy's SLSQP implementation; CasADi is an optional
cross-check and is never imported implicitly by the local path.
"""
from __future__ import annotations

import time as _clock

import numpy as np

from ..constraints.scalar import kde_safe_probability, kde_violation_upper
from ..kde.bandwidth import silverman_bandwidth


def transcribe_euler(problem, time, states, controls):
    """Return the Euler transcription and its defect and boundary residuals."""
    t = np.asarray(time, dtype=float).reshape(-1)
    X = np.asarray(states, dtype=float)
    U = np.asarray(controls, dtype=float).reshape(-1)
    if t.size < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("time must be a finite strictly increasing mesh")
    if X.shape != (t.size, 2) or U.size not in {t.size, t.size - 1}:
        raise ValueError("states must be (N,2), controls must contain N or N-1 values")
    interval_u = U[:-1] if U.size == t.size else U
    defects = np.asarray([
        X[k + 1] - X[k] - (t[k + 1] - t[k]) * problem.dynamics(t[k], X[k], interval_u[k])
        for k in range(t.size - 1)
    ])
    initial_residual = X[0] - np.asarray(problem.initial_state, dtype=float)
    terminal_residual = X[-1] - np.asarray(problem.terminal_state, dtype=float)
    return {
        "time": t, "states": X, "controls": U,
        "interval_controls": interval_u, "defects": defects,
        "initial_residual": initial_residual,
        "terminal_residual": terminal_residual,
        "n_state_variables": int(X.size), "n_control_variables": int(U.size),
    }


def _initial_guess(problem, t):
    """Construct a feasible-in-bounds warm start for the mesh dynamics."""
    u = np.full(t.size, min(problem.gravity, problem.u_max), dtype=float)
    X = np.empty((t.size, 2), dtype=float)
    X[0] = problem.initial_state
    for k, dt in enumerate(np.diff(t)):
        X[k + 1] = X[k] + dt * problem.dynamics(t[k], X[k], u[k])
    return X, u


def _samples(value, default=0.0):
    if value is None:
        return np.asarray([default], dtype=float)
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError("uncertainty samples must be nonempty and finite")
    return out


def _chance_constraints(problem, stage, terminal_samples, path_samples, epsilon_a,
                       epsilon_b, bandwidth=None):
    name = str(stage).lower().replace("-", "_")
    if name == "biased_kde":
        name = "local_shifted_epanechnikov"
    if name == "unbiased_kde":
        kernel = "gaussian"
    elif name == "local_shifted_epanechnikov":
        kernel = "epanechnikov"
    elif name in {"nominal", "mean", "worst_case", "worst", "scenario"}:
        kernel = None
    else:
        raise ValueError(f"unknown lunar continuation stage: {stage}")
    ht = silverman_bandwidth(terminal_samples) if bandwidth is None else float(bandwidth)
    hp = silverman_bandwidth(path_samples) if bandwidth is None else float(bandwidth)

    def values(states, controls):
        terminal = problem.terminal_residual(states[-1], terminal_samples)
        path = problem.path_residual_nodes(controls, path_samples)
        if name in {"nominal", "mean"}:
            terminal = problem.terminal_residual(states[-1], np.mean(terminal_samples))
            path = problem.path_residual_nodes(controls, np.mean(path_samples))
        if name in {"nominal", "mean", "worst_case", "worst"}:
            return np.asarray([-np.max(terminal), -np.max(path)], dtype=float)
        if name == "scenario":
            # Keep one inequality per sampled event.  This is deliberately
            # distinct from the worst-case reduction above.
            return np.r_[-terminal.reshape(-1), -path.reshape(-1)]
        path_samples_by_node = path.T
        if name == "unbiased_kde":
            terminal_safe = float(kde_safe_probability(terminal, ht, kernel))
            path_safe = np.asarray(kde_safe_probability(path_samples_by_node, hp, kernel), dtype=float)
        else:
            terminal_safe = 1.0 - float(kde_violation_upper(
                terminal, ht, kernel, bias=True))
            # The biased surrogate currently accepts vector samples only.  Apply
            # it independently at each node so the returned margins retain the
            # node dimension used by the NLP.
            path_safe = 1.0 - np.asarray([
                kde_violation_upper(path_samples_by_node[:, node], hp, kernel, bias=True)
                for node in range(path_samples_by_node.shape[1])
            ], dtype=float)
        return np.r_[terminal_safe - (1.0 - epsilon_a),
                     path_safe - (1.0 - epsilon_b)]

    return values, {
        "stage": name, "kernel": kernel,
        "terminal_bandwidth": float(ht), "path_bandwidth": float(hp),
        "terminal_sample_count": int(terminal_samples.size),
        "path_sample_count": int(path_samples.size),
        "epsilon_a": float(epsilon_a), "epsilon_b": float(epsilon_b),
        "local_biased_surrogate": name == "local_shifted_epanechnikov",
    }


def solve_collocation(problem, time=None, initial_states=None, initial_controls=None,
                      method="SLSQP", maxiter=300, tolerance=1e-6, use_casadi=False,
                      stage="nominal", terminal_samples=None, path_samples=None,
                      epsilon_a=None, epsilon_b=None, bandwidth=None):
    """Solve one local collocation stage and return auditable diagnostics."""
    if use_casadi:
        try:
            import casadi  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "CasADi/IPOPT is optional and unavailable; use the local SciPy backend"
            ) from exc
        raise NotImplementedError("CasADi cross-check is not enabled in this adapter")
    from scipy.optimize import minimize

    t = (np.linspace(0.0, problem.tf, 11) if time is None
         else np.asarray(time, dtype=float).reshape(-1))
    n = t.size
    X0, U0 = _initial_guess(problem, t)
    if initial_states is not None:
        X0 = np.asarray(initial_states, dtype=float).copy()
    if initial_controls is not None:
        U0 = np.asarray(initial_controls, dtype=float).reshape(-1).copy()
    if X0.shape != (n, 2) or U0.size not in {n, n - 1}:
        raise ValueError("initial states/controls have incompatible dimensions")
    if U0.size == n - 1:
        U0 = np.r_[U0, U0[-1]]

    ts, ps = _samples(terminal_samples), _samples(path_samples)
    ea = problem.epsilon_a if epsilon_a is None else float(epsilon_a)
    eb = problem.epsilon_b if epsilon_b is None else float(epsilon_b)
    chance, metadata = _chance_constraints(problem, stage, ts, ps, ea, eb, bandwidth)

    def unpack(z):
        return z[:2 * n].reshape(n, 2), z[2 * n:]

    def equality(z):
        X, U = unpack(z)
        tr = transcribe_euler(problem, t, X, U)
        # Position is governed by its chance event; only terminal velocity is
        # a deterministic boundary condition in the published benchmark.
        return np.r_[tr["defects"].reshape(-1), tr["initial_residual"],
                     tr["terminal_residual"][1]]

    def inequality(z):
        X, U = unpack(z)
        return chance(X, U)

    z0 = np.r_[X0.reshape(-1), U0]
    bounds = [(None, None)] * (2 * n) + [problem.control_bounds] * n
    started = _clock.perf_counter()
    result = minimize(
        lambda z: problem.objective(*unpack(z), t), z0, method=method,
        bounds=bounds,
        constraints=[{"type": "eq", "fun": equality},
                     {"type": "ineq", "fun": inequality}],
        options={"maxiter": int(maxiter), "ftol": float(tolerance)},
    )
    elapsed = _clock.perf_counter() - started
    X, U = unpack(result.x)
    eq_res, ineq_res = equality(result.x), inequality(result.x)
    return {
        "success": bool(result.success), "states": X, "controls": U, "time": t,
        "objective": float(result.fun), "nit": int(getattr(result, "nit", -1)),
        "message": str(result.message), "runtime_seconds": float(elapsed),
        "stage": metadata["stage"], "backend": "scipy_slsqp_euler",
        "max_defect": float(np.max(np.abs(eq_res))),
        "min_chance_margin": float(np.min(ineq_res)),
        "constraint_diagnostics": {
            **metadata, "max_defect": float(np.max(np.abs(eq_res))),
            "min_chance_margin": float(np.min(ineq_res)),
            "initial_residual": eq_res[-3:-1].tolist(),
            "terminal_velocity_residual": float(eq_res[-1]),
        },
    }


def solve_lunar_stages(problem, *, time=None, terminal_samples=None, path_samples=None,
                       stages=("nominal", "worst_case", "scenario", "unbiased_kde",
                               "local_shifted_epanechnikov"), maxiter=300,
                       tolerance=1e-6, epsilon_a=None, epsilon_b=None, bandwidth=None):
    """Run warm-start continuation and preserve failed stages."""
    ts, ps = _samples(terminal_samples), _samples(path_samples)
    records, states, controls = [], None, None
    last_successful_stage = None
    for index, stage in enumerate(stages):
        try:
            record = solve_collocation(
                problem, time=time, initial_states=states, initial_controls=controls,
                stage=stage, terminal_samples=ts, path_samples=ps, maxiter=maxiter,
                tolerance=tolerance, epsilon_a=epsilon_a, epsilon_b=epsilon_b,
                bandwidth=bandwidth,
            )
        except Exception as exc:  # preserve an auditable failed stage
            normalized = str(stage).lower().replace("-", "_")
            if normalized == "biased_kde":
                normalized = "local_shifted_epanechnikov"
            record = {
                "success": False,
                "stage": normalized,
                "backend": "scipy_slsqp_euler",
                "objective": None,
                "states": None,
                "controls": None,
                "time": None,
                "runtime_seconds": 0.0,
                "nit": -1,
                "message": f"{type(exc).__name__}: {exc}",
                "failure_type": type(exc).__name__,
            }
        record["stage_index"] = index
        record["warm_start_from"] = last_successful_stage
        records.append(record)
        if record["success"]:
            states, controls = record["states"], record["controls"]
            last_successful_stage = str(stage)
    result = {"success": bool(records and records[-1]["success"]),
              "stages": records, "final": records[-1] if records else None,
              "stage_order": [str(s) for s in stages],
              "backend": "scipy_slsqp_euler"}
    result.update({record["stage"]: record for record in records})
    return result


def solve_staged_collocation(problem, **kwargs):
    """Backward-compatible name for :func:`solve_lunar_stages`."""
    return solve_lunar_stages(problem, **kwargs)


solve_staged_lunar = solve_lunar_stages
transcribe = transcribe_euler
