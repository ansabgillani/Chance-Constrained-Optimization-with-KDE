"""Independent validation metrics and stable result records."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

SCHEMA_VERSION = "1.0"
REQUIRED_FIELDS = (
    "benchmark", "method", "seed", "n_train", "n_test", "epsilon",
    "objective", "train_violation", "test_violation", "estimated_violation",
    "gap", "runtime_seconds", "success", "message",
)


def _sample_events(residuals: np.ndarray) -> np.ndarray:
    """Return one Boolean event per Monte Carlo sample.

    The first dimension is the sample dimension. All remaining dimensions are
    constraint components; a sample violates the joint event when any component
    is positive.
    """
    values = np.asarray(residuals, dtype=float)
    if values.ndim == 0 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("residuals must be a non-empty finite array")
    if values.ndim == 1:
        return values > 0.0
    return np.any(values > 0.0, axis=tuple(range(1, values.ndim)))


def empirical_violation(residuals: np.ndarray) -> float:
    return float(np.mean(_sample_events(residuals)))


def conservatism_gap(epsilon: float, test_violation: float) -> float:
    epsilon, test_violation = float(epsilon), float(test_violation)
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0, 1]")
    if not 0 <= test_violation <= 1:
        raise ValueError("test_violation must lie in [0, 1]")
    return float(epsilon - test_violation)


def evaluate_residuals(
    train_residuals: np.ndarray,
    test_residuals: np.ndarray,
    *,
    epsilon: float,
    estimated_violation: float | None = None,
    objective: float | None = None,
    runtime_seconds: float | None = None,
    success: bool | None = None,
    message: str = "",
    benchmark: str | None = None,
    method: str | None = None,
    seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct one JSON-compatible validation record.

    ``train_residuals`` and ``test_residuals`` must use the first axis for
    independent samples. For vector constraints, all remaining components are
    jointly evaluated. ``estimated_violation`` is deliberately supplied by the
    estimator used during optimisation; it is not silently replaced by the
    empirical training value.
    """
    train_values = np.asarray(train_residuals, dtype=float)
    test_values = np.asarray(test_residuals, dtype=float)
    train_v, test_v = empirical_violation(train_values), empirical_violation(test_values)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "method": method,
        "seed": None if seed is None else int(seed),
        "epsilon": float(epsilon),
        "train_violation": train_v,
        "test_violation": test_v,
        "estimated_violation": None if estimated_violation is None else float(estimated_violation),
        "gap": conservatism_gap(epsilon, test_v),
        "objective": None if objective is None else float(objective),
        "runtime_seconds": None if runtime_seconds is None else float(runtime_seconds),
        "success": None if success is None else bool(success),
        "message": str(message),
        "n_train": int(train_values.shape[0]),
        "n_test": int(test_values.shape[0]),
    }
    if metadata:
        record.update(dict(metadata))
    return record
