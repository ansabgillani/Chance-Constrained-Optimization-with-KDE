"""Paper-aligned local sensitivity experiments.

This module is deliberately separate from the original evidence matrix.  It
implements the parts of the Schuster and Keil settings that are portable to
the local residual problems: a nested sample-size sequence, Schuster's
Gaussian bandwidth scaling for a scalar residual, and Keil's one-sided
Epanechnikov bias ``B(h)=h`` with bandwidth continuation.  IID NumPy samples,
SLSQP, and the finite-dimensional static model remain local choices; the
output metadata makes that distinction explicit.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..constraints import empirical_violation, kde_safe_probability, kde_violation_upper
from ..evaluation.io import write_array, write_run_bundle
from ..evaluation.metrics import evaluate_residuals
from ..kde.bandwidth import schuster_bandwidth
from ..problems import make_energy_dispatch, make_static_problem
from ..solvers import solve_dispatch, solve_static
from ..uncertainty import (sample_bimodal, sample_gaussian, sample_heavy_tailed,
                           sample_renewable_error, sample_skewed)


def _distribution(name: str, n: int, seed: int) -> np.ndarray:
    if name == "gaussian":
        return sample_gaussian(n, std=1.0, seed=seed)[0]
    if name == "bimodal":
        return sample_bimodal(n, means=(-1.2, 1.2), stds=(.35, .35), seed=seed)[0]
    if name == "skewed":
        return sample_skewed(n, scale=1.0, seed=seed)[0]
    if name == "heavy_tailed":
        return sample_heavy_tailed(n, df=3.5, scale=.8, seed=seed)[0]
    raise ValueError(name)


def _continuation(problem, initial, constraint_factory, solve, multipliers):
    """Solve from smooth/larger bandwidth to smaller bandwidth."""
    current = np.asarray(initial, dtype=float)
    records = []
    for multiplier in multipliers:
        result = solve(problem, initial_guess=current,
                       chance_constraint=constraint_factory(float(multiplier)),
                       maxiter=500)
        records.append((float(multiplier), result))
        current = result["decision"]
        if not result["success"]:
            break
    return records


def _static_rows(seed: int, sample_sizes: tuple[int, ...], n_test: int,
                 epsilon: float) -> tuple[list[dict], dict[str, np.ndarray]]:
    problem = make_static_problem()
    n_max = max(sample_sizes)
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for offset, name in enumerate(("gaussian", "bimodal", "skewed", "heavy_tailed")):
        stream_a = _distribution(name, n_max, seed + offset)
        stream_b = _distribution(name, n_max, seed + offset + 100)
        test_a = _distribution(name, n_test, seed + offset + 10000)
        test_b = _distribution(name, n_test, seed + offset + 10100)
        test = np.column_stack((test_a, test_b))
        arrays[f"static_{name}_test"] = test
        for n_train in sample_sizes:
            train = np.column_stack((stream_a[:n_train], stream_b[:n_train]))
            arrays[f"static_{name}_train_{n_train}"] = train
            scale_samples = train[:, 0]
            h = schuster_bandwidth(scale_samples, dimension=1)

            def residual(x, z):
                return np.asarray(problem.uncertain_residuals(x, z), float).reshape(-1)

            for method in ("schuster_gaussian", "keil_biased_epanechnikov"):
                if method == "schuster_gaussian":
                    multipliers = (1.0,)
                    estimator = lambda r, bw: 1.0 - float(kde_safe_probability(r, bw, "gaussian"))
                    def make_constraint(bw):
                        return lambda x: float((1.0 - epsilon) - kde_safe_probability(residual(x, train), bw, "gaussian"))
                else:
                    # Keil's B(h)=h shift is exact for the Epanechnikov kernel.
                    # The local analogue begins with a sharp surrogate and
                    # increases h until SLSQP can resolve the conservative
                    # constraint.  This is the Keil continuation direction;
                    # the mesh-error trigger is unavailable in this adapter.
                    multipliers = (0.25, 0.5, 1.0, 2.0)
                    estimator = lambda r, bw: float(kde_violation_upper(r, bw, "epanechnikov", True))
                    def make_constraint(bw):
                        return lambda x: float(kde_violation_upper(residual(x, train), bw, "epanechnikov", True) - epsilon)
                started = time.perf_counter()
                continuation = _continuation(problem, np.array([.5, .5]),
                                              lambda multiplier: make_constraint(h * multiplier),
                                              solve_static, multipliers)
                successful = [(multiplier, result) for multiplier, result in continuation
                              if result["success"]]
                selected_multiplier, solved = successful[-1] if successful else continuation[-1]
                x = solved["decision"]
                tr, te = residual(x, train), residual(x, test)
                selected_h = h * selected_multiplier
                row = evaluate_residuals(
                    tr, te, epsilon=epsilon,
                    estimated_violation=estimator(tr, selected_h),
                    objective=solved["objective"],
                    runtime_seconds=time.perf_counter() - started,
                    success=solved["success"], message=solved["message"],
                    benchmark="static_nonlinear_paper_aligned", method=method,
                    seed=seed + offset,
                    metadata={
                        "distribution": name, "bandwidth": selected_h,
                        "base_bandwidth": h, "bandwidth_rule": "schuster_gaussian_scaling",
                        "bias": "B(h)=h" if method.startswith("keil") else None,
                        "sample_schedule": list(sample_sizes), "continuation_multipliers": list(multipliers),
                        "selected_multiplier": selected_multiplier,
                        "continuation_attempts": [multiplier for multiplier, _ in continuation],
                        "continuation_success": [result["success"] for _, result in continuation],
                        "solver_iterations": solved["nit"],
                        "uncertainty_protocol": "IID synthetic stream; not Keil MCMC",
                    })
                rows.append(row)
    return rows, arrays


def _dispatch_rows(seed: int, sample_sizes: tuple[int, ...], n_test: int,
                   epsilon: float) -> tuple[list[dict], dict[str, np.ndarray]]:
    problem = make_energy_dispatch()
    n_max = max(sample_sizes)
    train_stream, meta = sample_renewable_error(n_max, seed=seed)
    test, _ = sample_renewable_error(n_test, seed=seed + 10000)
    rows: list[dict] = []
    arrays = {"dispatch_test": test}
    for n_train in sample_sizes:
        train = train_stream[:n_train]
        arrays[f"dispatch_train_{n_train}"] = train
        h = schuster_bandwidth(train, dimension=1)
        def residual(x, z):
            return problem.uncertain_residuals(x, z)[:, 0]
        for method in ("schuster_gaussian", "keil_biased_epanechnikov"):
            if method == "schuster_gaussian":
                multipliers = (1.0,)
                def make_constraint(bw):
                    return lambda x: float((1.0 - epsilon) - kde_safe_probability(residual(x, train), bw, "gaussian"))
                estimate = lambda r, bw: 1.0 - float(kde_safe_probability(r, bw, "gaussian"))
            else:
                multipliers = (0.25, 0.5, 1.0, 2.0)
                def make_constraint(bw):
                    return lambda x: float(kde_violation_upper(residual(x, train), bw, "epanechnikov", True) - epsilon)
                estimate = lambda r, bw: float(kde_violation_upper(r, bw, "epanechnikov", True))
            started = time.perf_counter()
            continuation = _continuation(problem, np.array([40., 40.]),
                                          lambda multiplier: make_constraint(h * multiplier),
                                          solve_dispatch, multipliers)
            successful = [(multiplier, result) for multiplier, result in continuation
                          if result["success"]]
            selected_multiplier, solved = successful[-1] if successful else continuation[-1]
            tr = residual(solved["decision"], train)
            te = residual(solved["decision"], test)
            rows.append(evaluate_residuals(
                tr, te, epsilon=epsilon,
                estimated_violation=estimate(tr, h * selected_multiplier),
                objective=solved["objective"], runtime_seconds=time.perf_counter() - started,
                success=solved["success"], message=solved["message"],
                benchmark="energy_dispatch_paper_aligned", method=method, seed=seed,
                metadata={"n_train_schedule": list(sample_sizes), "bandwidth": h * selected_multiplier,
                          "base_bandwidth": h, "bandwidth_rule": "schuster_gaussian_scaling",
                          "bias": "B(h)=h" if method.startswith("keil") else None,
                          "continuation_multipliers": list(multipliers),
                          "selected_multiplier": selected_multiplier,
                          "continuation_attempts": [multiplier for multiplier, _ in continuation],
                          "continuation_success": [result["success"] for _, result in continuation],
                          "uncertainty_metadata": meta,
                          "uncertainty_protocol": "IID synthetic stream; not Keil MCMC"}))
    return rows, arrays


def run_paper_aligned(output: str | Path = "results_paper_aligned", *,
                      seed: int = 20260906, force: bool = False,
                      sample_sizes: tuple[int, ...] = (250, 1000, 5000),
                      n_test: int = 10000) -> list[dict]:
    """Run the local, paper-aligned sensitivity matrix without overwriting prior results."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    static, arrays = _static_rows(seed, sample_sizes, n_test, .1)
    dispatch, dispatch_arrays = _dispatch_rows(seed + 1, sample_sizes, n_test, .1)
    arrays.update(dispatch_arrays)
    records = static + dispatch
    for name, values in arrays.items():
        write_array(values, root / "samples" / f"{name}.npy", force=force)
    metadata = {
        "profile": "paper_aligned",
        "seed": seed,
        "records": len(records),
        "sample_schedule": list(sample_sizes),
        "n_test": n_test,
        "epsilon": .1,
        "schuster": {
            "kernel": "Gaussian product; scalar residual d=1",
            "bandwidth": "s * (4/((d+2)N))**(1/(d+4))",
            "status": "scaling rule only; no finite-sample guarantee",
        },
        "keil": {
            "kernel": "biased Epanechnikov",
            "bias": "B(h)=h",
            "continuation": [0.25, 0.5, 1.0, 2.0],
            "sampling": "IID synthetic analogue, not 50,000-sample MCMC",
            "collocation": "not reproduced; local SLSQP static/dispatch only",
        },
    }
    write_run_bundle(records, metadata, root, force=force)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_paper_aligned")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_paper_aligned(args.output, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
