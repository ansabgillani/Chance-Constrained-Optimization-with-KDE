"""Bandwidth selection rules for one-dimensional KDEs."""

from __future__ import annotations

import numpy as np


def silverman_bandwidth(samples: np.ndarray, *, min_bandwidth: float = 1e-12) -> float:
    """Return Silverman's robust rule-of-thumb bandwidth.

    The baseline is ``0.9 * min(s, IQR / 1.34) * n**(-1/5)``.  If the
    robust scale vanishes, the available positive scale is used and a unit
    scale is used only for a constant sample.  ``min_bandwidth`` prevents a
    singular kernel in downstream optimization.
    """
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("samples must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must contain only finite values")
    floor = float(min_bandwidth)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("min_bandwidth must be positive and finite")
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    q25, q75 = np.percentile(x, [25.0, 75.0])
    iqr_scale = float((q75 - q25) / 1.34)
    if std > 0.0 and iqr_scale > 0.0:
        scale = min(std, iqr_scale)
    else:
        scale = max(std, iqr_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return max(floor, float(0.9 * scale * x.size ** (-0.2)))
