import csv
import json
from pathlib import Path


def test_saved_experiment_contract():
    root = Path(__file__).parents[1] / "results"
    summary = root / "summary.csv"
    metadata = root / "run_metadata.json"
    assert summary.exists() and summary.stat().st_size > 0
    assert metadata.exists()
    figure_root = root.parent / "kap09" / "figures"
    for name in ("fig_static_test_violation.pdf", "fig_static_tradeoff.pdf", "fig_sensitivity.pdf"):
        figure = figure_root / name
        assert figure.exists() and figure.stat().st_size > 1000
        assert figure.read_bytes().startswith(b"%PDF")
    rows = list(csv.DictReader(summary.open()))
    assert rows
    assert {"benchmark", "method", "seed", "test_violation", "objective"}.issubset(rows[0])
    payload = json.loads(metadata.read_text())
    assert payload["seed"] == 20260906
