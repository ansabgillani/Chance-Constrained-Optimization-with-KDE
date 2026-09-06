"""Solver adapters.

Finite-dimensional static and dispatch adapters are required dependencies.
The lunar collocation adapter is optional until Task 7 is installed, so a
clean checkout of the static solver remains importable without CasADi or the
collocation module.
"""

from .static import SolverResult, solve, solve_dispatch, solve_static, solve_static_problem

__all__ = ["SolverResult", "solve_static", "solve_static_problem", "solve_dispatch", "solve"]

try:  # optional module; Task 7 supplies it in the complete package
    from .collocation import solve_collocation, transcribe, transcribe_euler
except ImportError:  # pragma: no cover - exercised only in a Task 6-only checkout
    pass
else:
    __all__ += ["transcribe_euler", "transcribe", "solve_collocation"]
