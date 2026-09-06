"""Metrics and provenance-preserving result serialization."""
from .metrics import REQUIRED_FIELDS, SCHEMA_VERSION, conservatism_gap, empirical_violation, evaluate_residuals
from .io import write_array, write_result, write_results_csv, write_run_bundle

__all__ = ["REQUIRED_FIELDS", "SCHEMA_VERSION", "empirical_violation", "evaluate_residuals",
           "conservatism_gap", "write_array", "write_result", "write_results_csv", "write_run_bundle"]
