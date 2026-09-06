"""Reproducible synthetic uncertainty samplers.

Each function returns ``(samples, metadata)``.  Sampling is deliberately
isolated from NumPy's legacy global random state: a fresh
``numpy.random.Generator`` is created from the integer ``seed`` on every call.
The metadata is JSON-compatible so that the exact distribution used by an
experiment can be recorded with its result.
"""

from __future__ import annotations

import numpy as np


def _generator(seed: int | None) -> np.random.Generator:
    """Create the sole source of randomness used by this module.

    An integer seed (or ``None`` for explicitly requested nondeterminism) is
    accepted.  Rejecting booleans and arbitrary objects avoids silently
    converting an accidental configuration value into a different seed.
    """
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer or None, not bool")
    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or None")
    return np.random.default_rng(None if seed is None else int(seed))


def _size(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    return int(n)


def _metadata_seed(seed: int | None) -> int | None:
    """Return a JSON-compatible representation of ``seed``."""
    return None if seed is None else int(seed)


def sample_gaussian(n: int, *, mean: float = 0.0, std: float = 1.0,
                    seed: int | None = None) -> tuple[np.ndarray, dict]:
    n = _size(n)
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("mean must be finite and std must be positive")
    samples = _generator(seed).normal(mean, std, n)
    return samples, {"distribution": "gaussian", "mean": float(mean), "std": float(std), "seed": _metadata_seed(seed)}


def sample_bimodal(n: int, *, means: tuple[float, float] = (-2.0, 2.0),
                   stds: tuple[float, float] = (0.5, 0.5),
                   weights: tuple[float, float] = (0.5, 0.5),
                   seed: int | None = None) -> tuple[np.ndarray, dict]:
    n = _size(n)
    try:
        n_means, n_stds, n_weights = len(means), len(stds), len(weights)
    except TypeError as exc:
        raise ValueError("bimodal parameters must be length-two sequences") from exc
    if n_means != 2 or n_stds != 2 or n_weights != 2:
        raise ValueError("bimodal parameters must have length two")
    means = tuple(float(v) for v in means)
    stds = tuple(float(v) for v in stds)
    weights = tuple(float(v) for v in weights)
    if any(not np.isfinite(v) for v in (*means, *stds, *weights)) or any(v <= 0 for v in stds):
        raise ValueError("bimodal parameters must be finite and stds positive")
    if any(v < 0 for v in weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    rng = _generator(seed)
    component = rng.choice(2, size=n, p=weights)
    samples = rng.normal(np.asarray(means)[component], np.asarray(stds)[component])
    metadata = {"distribution": "bimodal", "means": list(map(float, means)),
                "stds": list(map(float, stds)), "weights": list(map(float, weights)),
                "seed": _metadata_seed(seed)}
    return samples, metadata


def sample_skewed(n: int, *, scale: float = 1.0, seed: int | None = None) -> tuple[np.ndarray, dict]:
    n = _size(n)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    samples = _generator(seed).exponential(scale, n) - scale
    return samples, {"distribution": "skewed_exponential", "scale": float(scale), "seed": _metadata_seed(seed)}


def sample_heavy_tailed(n: int, *, df: float = 3.0, scale: float = 1.0,
                        seed: int | None = None) -> tuple[np.ndarray, dict]:
    n = _size(n)
    if not np.isfinite(df) or df <= 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("df must be positive and scale must be positive and finite")
    samples = _generator(seed).standard_t(df, n) * scale
    return samples, {"distribution": "student_t", "df": float(df), "scale": float(scale), "seed": _metadata_seed(seed)}


def sample_renewable_error(n: int, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Sample a fixed two-mode renewable forecast error mixture.

    The canonical synthetic model has a 60% under-production mode (mean
    ``-0.25``, standard deviation ``0.10``) and a 40% over-production mode
    (mean ``+0.35``, standard deviation ``0.15``), in normalized power units.
    The negative mode represents a renewable shortfall and the positive mode
    represents excess generation.  This is a declared synthetic distribution;
    it is not fitted to an external time series.
    """
    samples, meta = sample_bimodal(
        n, means=(-0.25, 0.35), stds=(0.10, 0.15), weights=(0.6, 0.4), seed=seed
    )
    meta["distribution"] = "renewable_error"
    meta["mixture_components"] = [
        {"label": "underproduction", "weight": 0.6, "mean": -0.25, "std": 0.10},
        {"label": "overproduction", "weight": 0.4, "mean": 0.35, "std": 0.15},
    ]
    meta["units"] = "normalised power error"
    return samples, meta
