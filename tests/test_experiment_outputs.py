import csv
import json
from pathlib import Path


def test_saved_experiment_contract():
    root = Path(__file__).parents[1] / "results"
    summary = root / "summary.csv"
    metadata = root / "run_metadata.json"
    assert summary.exists() and summary.stat().st_size > 0
    assert metadata.exists()
    figure = root / "kap08" / "figures" / "fig_static_test_violation.pdf"
    assert figure.exists() and figure.read_bytes().startswith(b"%PDF")
    rows = list(csv.DictReader(summary.open()))
    assert rows
    assert {"benchmark", "method", "seed", "test_violation", "objective"}.issubset(rows[0])
    payload = json.loads(metadata.read_text())
    assert payload["seed"] == 20260906
