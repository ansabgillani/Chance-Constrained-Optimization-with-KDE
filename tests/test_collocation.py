import numpy as np
import pytest

from kde_cco.problems.lunar_landing import make_lunar_landing
from kde_cco.solvers.collocation import (
    _chance_constraints,
    solve_collocation,
    solve_lunar_stages,
    transcribe_euler,
)


def test_euler_transcription_dimensions_and_defects():
    problem = make_lunar_landing()
    t = np.linspace(0.0, problem.tf, 5)
    states = np.zeros((5, 2))
    controls = np.zeros(5)
    result = transcribe_euler(problem, t, states, controls)
    assert result["defects"].shape == (4, 2)
    assert result["n_state_variables"] == 10
    assert result["n_control_variables"] == 5
    assert np.allclose(result["defects"][:, 0], 0.0)


def test_euler_defect_for_constant_acceleration():
    problem = make_lunar_landing()
    t = np.array([0.0, 1.0, 2.0])
    controls = np.full(3, problem.gravity)
    states = np.array([[100.0, 0.0], [100.0, 0.0], [100.0, 0.0]])
    result = transcribe_euler(problem, t, states, controls)
    assert np.allclose(result["defects"], 0.0)


def test_local_scipy_stage_has_integral_objective_and_boundaries():
    problem = make_lunar_landing()
    result = solve_collocation(
        problem,
        stage="nominal",
        terminal_samples=np.array([0.0]),
        path_samples=np.array([0.0]),
        maxiter=250,
    )
    assert result["backend"] == "scipy_slsqp_euler"
    assert result["states"].shape == (11, 2)
    assert result["controls"].shape == (11,)
    assert np.all(result["controls"] >= -1e-8)
    assert np.all(result["controls"] <= problem.u_max + 1e-8)
    assert np.isfinite(result["objective"])
    assert result["objective"] >= 0.0
    assert result["max_defect"] < 1e-5


def test_staged_solver_preserves_order_and_warm_starts():
    problem = make_lunar_landing()
    result = solve_lunar_stages(
        problem,
        terminal_samples=np.array([-0.1, 0.1]),
        path_samples=np.array([-0.1, 0.1]),
        stages=("nominal", "worst_case", "unbiased_kde", "local_shifted_epanechnikov"),
        maxiter=80,
    )
    assert result["stage_order"] == [
        "nominal", "worst_case", "unbiased_kde", "local_shifted_epanechnikov"
    ]
    assert len(result["stages"]) == 4
    assert result["stages"][0]["warm_start_from"] is None
    assert result["stages"][1]["warm_start_from"] == "nominal"
    assert "local_shifted_epanechnikov" in result
    assert all("message" in record for record in result["stages"])


def test_biased_path_surrogate_returns_one_margin_per_node():
    problem = make_lunar_landing()
    constraints, _ = _chance_constraints(
        problem, "local_shifted_epanechnikov", np.array([-0.1, 0.1]),
        np.array([-0.1, 0.1]), problem.epsilon_a, problem.epsilon_b,
    )
    states = np.array([[100.0, 0.0], [0.0, 0.0]])
    margins = constraints(states, np.array([1.0, 1.0]))
    assert margins.shape == (3,)
    assert np.all(np.isfinite(margins))


def test_casadi_path_is_explicit_when_unavailable():
    problem = make_lunar_landing()
    try:
        import casadi  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="optional and unavailable"):
            solve_collocation(problem, use_casadi=True)
