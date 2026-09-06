import numpy as np

from kde_cco.problems import make_energy_dispatch, make_static_problem
from kde_cco.solvers import solve_dispatch, solve_static


def _assert_schema(result, n):
    assert result["success"]
    assert result["decision"].shape == (n,)
    assert np.isfinite(result["objective"])
    assert result["nit"] >= 0
    assert result["message"]
    assert result["runtime_seconds"] >= 0.0


def test_static_solver_smoke_and_constraint_sign():
    problem = make_static_problem()
    samples = np.zeros((32, 2))
    # Residual <= 0 is the package convention; the adapter flips the sign.
    constraint = {"constraint": lambda x: float(np.max(problem.uncertain_residuals(x, samples)))}
    result = solve_static(problem, initial_guess=np.array([0.2, 0.2]),
                          chance_constraint=constraint, maxiter=300)
    _assert_schema(result, 2)
    assert np.max(problem.uncertain_residuals(result["decision"], samples)) <= 1e-6


def test_dispatch_adapter_optimises_feasible_two_generator_dispatch():
    problem = make_energy_dispatch()
    result = solve_dispatch(problem, initial_guess=np.array([40.0, 40.0]), maxiter=300)
    _assert_schema(result, 2)
    x = result["decision"]
    assert np.all(x >= -1e-7)
    assert x[0] <= problem.capacities[0] + 1e-7
    assert x[1] <= problem.capacities[1] + 1e-7
    # Reserve is zero by default, so deterministic balance must be met.
    assert x.sum() >= problem.demand - problem.renewable_nominal - 1e-6
