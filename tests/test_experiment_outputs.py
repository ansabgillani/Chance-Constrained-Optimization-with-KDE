import csv
import json
from pathlib import Path


def test_saved_experiment_contract():
    root = Path(__file__).parents[1] / "results"
    summary = root / "summary.csv"
    metadata = root / "run_metadata.json"
    assert summary.exists() and summary.stat().st_size > 0
    assert metadata.exists()
    rows = list(csv.DictReader(summary.open()))
    assert rows
    assert {"benchmark", "method", "seed", "test_violation", "objective"}.issubset(rows[0])
    payload = json.loads(metadata.read_text())
    assert payload["seed"] == 20260906
