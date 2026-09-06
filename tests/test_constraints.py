import numpy as np
import pytest
from kde_cco.constraints.scalar import (
    residuals, empirical_violation, kde_safe_probability, kde_violation_upper,
    kde_safe_probability_gradient,
)
from kde_cco.kde.kernels import biased_epanechnikov_cdf
from kde_cco.constraints.joint import boole_risk_allocation, max_violation, smooth_max


def test_scalar_residual_and_empirical_violation():
    out = residuals(np.array([1.0, 2.0]), np.array([-1.0, 0.5]))
    assert np.array_equal(out, np.array([0.0, 2.5]))
    assert empirical_violation(np.array([-1.0, 0.0, 0.2])) == pytest.approx(1 / 3)


def test_matrix_preserves_sample_axis():
    matrix = np.array([[-1.0, 1.0], [1.0, -1.0], [-2.0, 2.0]])
    assert np.allclose(empirical_violation(matrix), [1 / 3, 2 / 3])
    safe = kde_safe_probability(matrix, 0.2)
    assert safe.shape == (2,)


def test_kde_safe_and_biased_violation_are_conservative():
    r = np.array([-1.0, 0.0, 1.0])
    safe = kde_safe_probability(r, 0.25, kernel="epanechnikov")
    upper = kde_violation_upper(r, 0.25, kernel="epanechnikov", bias=True)
    assert 0 <= safe <= 1
    assert upper >= empirical_violation(r)


def test_biased_surrogate_dominates_strict_violation_pointwise():
    # The residual convention is strict for violations (r > 0); r == 0 is
    # safe.  The translated kernel returns zero safe mass for all r > 0.
    r = np.linspace(-2.0, 2.0, 401)
    h = 0.31
    violation_surrogate = 1.0 - biased_epanechnikov_cdf(-r / h)
    strict_indicator = (r > 0.0).astype(float)
    assert np.all(violation_surrogate + 1e-14 >= strict_indicator)


def test_joint_constraints():
    assert np.allclose(boole_risk_allocation(0.1, 4), np.full(4, 0.025))
    matrix = np.array([[-1, 0.2], [0.1, -2]])
    assert max_violation(matrix) == pytest.approx(0.2)
    z = np.array([1.0, 2.0, 3.0])
    assert smooth_max(z, 0.5) >= np.max(z)


def test_gaussian_kde_chain_gradient_matches_central_difference():
    uncertainty = np.array([-0.7, -0.1, 0.25, 0.9])
    x = 0.18
    h = 0.37
    values = x + uncertainty
    analytic = kde_safe_probability_gradient(values, np.ones(values.size), h)
    step = 1e-6
    finite_difference = (
        kde_safe_probability(values + step, h)
        - kde_safe_probability(values - step, h)
    ) / (2.0 * step)
    assert analytic == pytest.approx(finite_difference, rel=1e-5, abs=1e-7)


def test_gradient_shape_validation():
    with pytest.raises(ValueError):
        kde_safe_probability_gradient(np.array([0.0, 1.0]), np.ones(3), 0.2)
    with pytest.raises(ValueError):
        kde_safe_probability_gradient(np.zeros((2, 2)), np.ones((2, 1)), 0.2)
