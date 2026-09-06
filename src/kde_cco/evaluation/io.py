"""Safe serialization for experiment results.

Writers never overwrite an existing path unless ``force=True`` is explicitly
provided. This protects prior generated evidence while allowing deliberate
reproduction of a run.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        return _json_default(value)
    except TypeError:
        return value


def write_result(record: Mapping[str, Any], path: str | Path, *, force: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing result: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(dict(record)), indent=2, sort_keys=True) + "\n")
    return target


def write_array(values: Any, path: str | Path, *, force: bool = False) -> Path:
    import numpy as np
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing array: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.asarray(values))
    return target


def write_results_csv(records: Iterable[Mapping[str, Any]], path: str | Path, *, force: bool = False) -> Path:
    rows = [dict(r) for r in records]
    if not rows:
        raise ValueError("records must not be empty")
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing result: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    preferred = ["benchmark", "method", "seed", "n_train", "n_test", "epsilon",
                 "objective", "train_violation", "test_violation", "estimated_violation",
                 "gap", "runtime_seconds", "success", "message"]
    all_fields = {key for row in rows for key in row}
    fields = [key for key in preferred if key in all_fields]
    fields.extend(sorted(all_fields.difference(fields)))
    def csv_value(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(_json_safe(value), sort_keys=True)
        return _json_safe(value)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})
    return target


def write_run_bundle(records: Iterable[Mapping[str, Any]], metadata: Mapping[str, Any],
                     output: str | Path, *, force: bool = False) -> Path:
    """Write per-record JSON, summary CSV, metadata, and optional diagnostics.

    Existing files are checked individually. No deletion or directory cleanup is
    performed, even with ``force``.
    """
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows = [dict(r) for r in records]
    if not rows:
        raise ValueError("records must not be empty")
    for index, row in enumerate(rows):
        write_result(row, root / f"record_{index:04d}.json", force=force)
    write_results_csv(rows, root / "summary.csv", force=force)
    write_result(metadata, root / "run_metadata.json", force=force)
    return root
