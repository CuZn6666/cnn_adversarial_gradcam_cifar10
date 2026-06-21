import json

import pytest

from src.metrics import load_metrics, save_metrics


def _single_metrics() -> dict[str, float | int]:
    return {
        "mean_loss": 1.25,
        "accuracy": 0.5,
        "total_samples": 4,
        "epoch": 1,
        "learning_rate": 0.001,
        "batch_size": 2,
        "seed": 42,
    }


def test_save_and_load_metrics_dictionary(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics = _single_metrics()

    saved_path = save_metrics(metrics, metrics_path)
    loaded_metrics = load_metrics(metrics_path)

    assert saved_path == metrics_path
    assert metrics_path.is_file()
    assert loaded_metrics == metrics


def test_save_and_load_epoch_metrics_list(tmp_path) -> None:
    metrics_path = tmp_path / "history.json"
    metrics = [
        _single_metrics(),
        {
            **_single_metrics(),
            "epoch": 2,
            "mean_loss": 0.9,
            "accuracy": 0.75,
        },
    ]

    save_metrics(metrics, metrics_path)

    assert load_metrics(metrics_path) == metrics


def test_save_metrics_creates_parent_directories(tmp_path) -> None:
    metrics_path = tmp_path / "nested" / "results" / "metrics.json"

    save_metrics(_single_metrics(), metrics_path)

    assert metrics_path.is_file()


def test_save_metrics_rejects_non_json_serializable_values(tmp_path) -> None:
    metrics_path = tmp_path / "invalid.json"
    metrics = {"values": {1, 2, 3}}

    with pytest.raises(
        ValueError,
        match="Metrics must be JSON-serializable",
    ):
        save_metrics(metrics, metrics_path)

    assert not metrics_path.exists()


def test_saved_metrics_json_is_deterministic_and_human_readable(
    tmp_path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    metrics = {
        "seed": 42,
        "accuracy": 0.5,
        "mean_loss": 1.25,
    }

    save_metrics(metrics, first_path)
    save_metrics(metrics, second_path)

    expected_text = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    assert first_path.read_text(encoding="utf-8") == expected_text
    assert second_path.read_text(encoding="utf-8") == expected_text
