import numpy as np
import pytest

from kde_cco.baselines.methods import gaussian_parametric
from kde_cco.problems.energy_dispatch import make_energy_dispatch
from kde_cco.problems.lunar_landing import make_lunar_landing
from kde_cco.problems.static_problem import make_static_problem


def test_static_has_explicit_dimensions_and_residual_convention():
    problem = make_static_problem(noise_scale=0.2, radius=1.1)
    assert problem.n_variables == 2
    assert problem.uncertainty_dim == 2
    assert problem.n_uncertain_constraints == 1
    assert problem.metadata["residual_convention"] == "residual <= 0 is safe"
    assert problem.metadata["noise_scale"] == pytest.approx(0.2)
    assert problem.metadata["radius"] == pytest.approx(1.1)
    assert problem.uncertain_residuals(np.zeros(2), np.zeros((7, 2))).shape == (7,)


def test_energy_dispatch_uses_scalar_error_and_three_residual_components():
    problem = make_energy_dispatch(demand=100, renewable_nominal=30, capacities=(90, 100))
    assert problem.n_variables == 2
    assert problem.uncertainty_dim == 1
    assert problem.n_uncertain_constraints == 3
    assert problem.metadata["residual_convention"] == "residual <= 0 is safe"
    residual = problem.uncertain_residuals(np.array([40.0, 35.0]), np.zeros((5, 1)))
    assert residual.shape == (5, 3)
    assert np.all(residual[:, 0] <= 0)
    assert np.all(residual[:, 1:] <= 0)
    assert problem.balance_residual(np.array([40.0, 35.0]), np.array([0.0, 5.0])).shape == (2,)


def test_lunar_residuals_match_published_terminal_and_path_events():
    problem = make_lunar_landing()
    assert problem.u_max == pytest.approx(3.0)
    assert problem.uncertainty_dim == 1
    assert problem.n_terminal_constraints == 1
    assert problem.n_path_constraints == 1
    assert problem.metadata["terminal_sigma"] == pytest.approx(0.1)
    xi = np.array([[-0.1], [0.0], [0.2]])
    terminal = problem.terminal_residual(np.array([0.0, 0.0]), xi)
    assert terminal.shape == (3,)
    assert np.allclose(terminal, np.abs(-xi[:, 0]) - problem.delta)
    path = problem.path_residual(2.5, xi)
    assert path.shape == (3,)
    assert np.allclose(path, 2.5 + xi[:, 0] - problem.u_max)


def test_gaussian_baseline_refuses_nonlinear_residuals():
    problem = make_static_problem()
    samples = np.zeros((16, 2))
    with pytest.raises(NotImplementedError, match="affine"):
        gaussian_parametric(problem.uncertain_residuals, samples)


def test_gaussian_baseline_accepts_scalar_affine_uncertainty():
    samples = np.linspace(-1.0, 1.0, 21)

    def affine_residual(x, xi):
        return np.asarray(x)[0] + np.asarray(xi).reshape(-1)

    baseline = gaussian_parametric(affine_residual, samples, epsilon=0.05)
    assert baseline["metadata"]["uncertainty_model"] == "scalar_affine_gaussian"
    assert np.isfinite(baseline["constraint"](np.array([0.0])))
