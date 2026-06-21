from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Metrics = dict[str, Any] | list[dict[str, Any]]


def save_metrics(metrics: Metrics, path: str | Path) -> Path:
    """Save metrics as deterministic, human-readable JSON."""
    try:
        serialized_metrics = json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Metrics must be JSON-serializable.") from error

    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(serialized_metrics + "\n", encoding="utf-8")
    return metrics_path


def load_metrics(path: str | Path) -> Metrics:
    """Load metrics from a JSON file."""
    metrics_path = Path(path)
    with metrics_path.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)
