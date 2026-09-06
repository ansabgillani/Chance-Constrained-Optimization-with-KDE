import numpy as np
from kde_cco.problems.static_problem import make_static_problem
from kde_cco.solvers.static import solve_static

def test_static_solver_smoke():
    p = make_static_problem()
    result = solve_static(p, initial_guess=np.array([0.,0.]), maxiter=100)
    assert result['success']
    assert result['decision'].shape == (2,)
    assert np.isfinite(result['objective'])
