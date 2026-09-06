import numpy as np
from kde_cco.problems.static_problem import make_static_problem
from kde_cco.problems.energy_dispatch import make_energy_dispatch
from kde_cco.problems.lunar_landing import make_lunar_landing

def test_static_shapes_and_bounds():
    p = make_static_problem()
    assert p.n_variables == 2
    assert np.asarray(p.objective(np.zeros(2))).ndim == 0
    assert np.asarray(p.uncertain_residuals(np.zeros(2), np.zeros((4,2)))).shape == (4,)

def test_energy_balance_and_capacity():
    p = make_energy_dispatch()
    x = np.array([30., 40.])
    assert p.n_variables == 2
    assert np.asarray(p.balance_residual(x, np.array([0., 5.]))).shape == (2,)
    assert np.all(np.asarray(p.capacity_residual(x)) <= 0)

def test_lunar_dynamics_and_residuals():
    p = make_lunar_landing()
    y = np.array([10., 2.])
    assert np.allclose(p.dynamics(0., y, 0.), [2., -1.622])
    assert p.u_max > 0
    assert np.asarray(p.terminal_residual(np.array([0., 0.]), np.array([0., 0.]))).shape == (2,)
    assert np.asarray(p.path_residual(np.array([0., p.u_max]), np.array([0., 0.]))).shape == (2,)
