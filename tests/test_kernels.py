import numpy as np
from scipy.integrate import quad
from kde_cco.kde.kernels import (
    gaussian_pdf, epanechnikov_pdf, gaussian_cdf, epanechnikov_cdf,
    biased_epanechnikov_cdf,
)
from kde_cco.kde.bandwidth import silverman_bandwidth


def test_kernel_densities_normalize():
    assert np.isclose(quad(lambda z: float(gaussian_pdf(z)), -np.inf, np.inf)[0], 1.0)
    assert np.isclose(quad(lambda z: float(epanechnikov_pdf(z)), -1, 1)[0], 1.0)


def test_scaled_kernels_normalize_for_nonunit_bandwidth():
    h = 0.37
    assert np.isclose(quad(lambda z: float(gaussian_pdf(z, h)), -np.inf, np.inf)[0], 1.0)
    assert np.isclose(quad(lambda z: float(epanechnikov_pdf(z, h)), -h, h)[0], 1.0)


def test_integrated_kernels_have_limits():
    assert gaussian_cdf(np.array([-12.0]))[0] < 1e-20
    assert gaussian_cdf(np.array([12.0]))[0] > 1 - 1e-20
    assert np.all(epanechnikov_cdf(np.array([-2.0, 2.0])) == [0.0, 1.0])
    assert np.all(biased_epanechnikov_cdf(np.array([-1.0, 3.0])) == [0.0, 1.0])


def test_integrated_kernels_are_monotone_and_bounded():
    x = np.linspace(-3.0, 3.0, 1001)
    for cdf in (gaussian_cdf, epanechnikov_cdf, biased_epanechnikov_cdf):
        values = cdf(x)
        assert np.all((values >= 0.0) & (values <= 1.0))
        assert np.all(np.diff(values) >= -1e-14)


def test_biased_kernel_is_conservative_for_positive_residuals():
    # Standardized residual r > 0 is a violation; biased safe contribution vanishes.
    r = np.array([0.1, 0.5, 2.0])
    assert np.all(biased_epanechnikov_cdf(-r) <= 0.0)


def test_biased_kernel_dominates_the_violation_indicator_pointwise():
    residual = np.linspace(-2.0, 2.0, 401)
    safe_indicator = residual <= 0.0
    surrogate = biased_epanechnikov_cdf(-residual)
    assert np.all(surrogate <= safe_indicator.astype(float) + 1e-15)
    assert np.all((surrogate >= 0.0) & (surrogate <= 1.0))


def test_silverman_bandwidth_is_positive_and_validates():
    samples = np.linspace(-1, 2, 31)
    h = silverman_bandwidth(samples)
    assert np.isfinite(h) and h > 0


def test_silverman_validates_empty_nonfinite_and_floor():
    import pytest
    with pytest.raises(ValueError):
        silverman_bandwidth(np.array([]))
    with pytest.raises(ValueError):
        silverman_bandwidth(np.array([0.0, np.nan]))
    assert silverman_bandwidth(np.ones(10), min_bandwidth=0.25) >= 0.25
