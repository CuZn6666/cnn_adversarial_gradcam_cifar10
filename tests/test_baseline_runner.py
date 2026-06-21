from pathlib import Path

import numpy as np

from configs.default_config import BaselineConfig
from experiments.baseline.train_baseline import run_synthetic_baseline
from src.metrics import load_metrics


def _tiny_config(tmp_path: Path) -> BaselineConfig:
    return BaselineConfig(
        learning_rate=5e-4,
        batch_size=2,
        epochs=2,
        seed=17,
        train_subset_size=4,
        eval_subset_size=3,
        checkpoint_dir=tmp_path / "unused_checkpoints",
        log_dir=tmp_path / "unused_logs",
        figure_dir=tmp_path / "unused_figures",
    )


def test_synthetic_baseline_runner_creates_artifacts_and_metrics(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "synthetic_run"
    result = run_synthetic_baseline(
        config=_tiny_config(tmp_path),
        output_root=output_root,
    )

    artifact_keys = (
        "checkpoint_path",
        "metrics_path",
        "loss_curve_path",
        "accuracy_curve_path",
    )
    for key in artifact_keys:
        path = result[key]
        assert isinstance(path, Path)
        assert path.is_file()
        assert path.is_relative_to(output_root)

    assert result["loss_curve_path"].read_bytes().startswith(b"\x89PNG")
    assert result["accuracy_curve_path"].read_bytes().startswith(b"\x89PNG")

    metrics_history = load_metrics(result["metrics_path"])
    assert metrics_history == result["metrics_history"]
    assert len(metrics_history) == 2

    expected_fields = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_total_samples",
        "eval_loss",
        "eval_accuracy",
        "eval_total_samples",
        "learning_rate",
        "batch_size",
        "seed",
    }
    for metrics in metrics_history:
        assert expected_fields <= metrics.keys()
        assert np.isfinite(metrics["train_loss"])
        assert np.isfinite(metrics["eval_loss"])
        assert 0.0 <= metrics["train_accuracy"] <= 1.0
        assert 0.0 <= metrics["eval_accuracy"] <= 1.0
        assert metrics["train_total_samples"] == 4
        assert metrics["eval_total_samples"] == 3

    assert result["final_metrics"] == metrics_history[-1]


def test_synthetic_baseline_runner_is_deterministic(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    first = run_synthetic_baseline(config, tmp_path / "first")
    second = run_synthetic_baseline(config, tmp_path / "second")

    assert first["metrics_history"] == second["metrics_history"]
    assert first["final_metrics"] == second["final_metrics"]

    with np.load(first["checkpoint_path"]) as first_checkpoint:
        with np.load(second["checkpoint_path"]) as second_checkpoint:
            assert first_checkpoint.files == second_checkpoint.files
            for key in first_checkpoint.files:
                np.testing.assert_array_equal(
                    first_checkpoint[key],
                    second_checkpoint[key],
                )
