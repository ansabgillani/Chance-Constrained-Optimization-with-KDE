import json
import numpy as np
import pytest

from kde_cco.evaluation.metrics import evaluate_residuals
from kde_cco.evaluation.io import write_result


def test_metrics_and_gap_are_exact():
    result = evaluate_residuals(
        np.array([-1.0, 1.0]), np.array([-2.0, 0.5, 0.7]), epsilon=0.2,
        metadata={"seed": 4},
    )
    assert result["train_violation"] == pytest.approx(0.5)
    assert result["test_violation"] == pytest.approx(2 / 3)
    assert result["gap"] == pytest.approx(0.2 - 2 / 3)


def test_result_writer_refuses_overwrite(tmp_path):
    target = tmp_path / "run.json"
    write_result({"objective": 1.0}, target)
    with pytest.raises(FileExistsError):
        write_result({"objective": 2.0}, target)
    assert json.loads(target.read_text())["objective"] == 1.0
