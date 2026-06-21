from pathlib import Path

import numpy as np
import pytest

from configs.default_config import CIFAR10_EXTRACTED_DIR, BaselineConfig
from experiments.baseline import train_baseline
from src.metrics import load_metrics


def _tiny_config(tmp_path: Path) -> BaselineConfig:
    return BaselineConfig(
        learning_rate=5e-4,
        batch_size=2,
        epochs=1,
        seed=23,
        train_subset_size=4,
        eval_subset_size=3,
        checkpoint_dir=tmp_path / "unused_checkpoints",
        log_dir=tmp_path / "unused_logs",
        figure_dir=tmp_path / "unused_figures",
    )


def _fake_cifar10() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    rng = np.random.default_rng(101)
    train_images = rng.random((8, 3, 32, 32), dtype=np.float32)
    train_labels = np.arange(8, dtype=np.int64) % 10
    eval_images = rng.random((6, 3, 32, 32), dtype=np.float32)
    eval_labels = np.arange(6, dtype=np.int64) % 10
    class_names = [f"class_{index}" for index in range(10)]
    return (
        train_images,
        train_labels,
        eval_images,
        eval_labels,
        class_names,
    )


def test_cifar10_subset_runner_requires_existing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded(data_dir: Path) -> tuple:
        raise AssertionError(f"Unexpected loader call for {data_dir}.")

    monkeypatch.setattr(train_baseline, "load_cifar10", fail_if_loaded)

    with pytest.raises(
        FileNotFoundError,
        match="Download and extract it explicitly",
    ):
        train_baseline.run_cifar10_subset_baseline(
            config=_tiny_config(tmp_path),
            output_root=tmp_path / "output",
            data_dir=tmp_path / "missing_data",
        )


def test_cifar10_subset_runner_creates_deterministic_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    (data_dir / CIFAR10_EXTRACTED_DIR).mkdir(parents=True)
    monkeypatch.setattr(
        train_baseline,
        "load_cifar10",
        lambda data_path: _fake_cifar10(),
    )
    config = _tiny_config(tmp_path)

    first = train_baseline.run_cifar10_subset_baseline(
        config,
        tmp_path / "first",
        data_dir,
    )
    second = train_baseline.run_cifar10_subset_baseline(
        config,
        tmp_path / "second",
        data_dir,
    )

    expected_names = {
        "checkpoint_path": "cifar10_subset_baseline.npz",
        "metrics_path": "cifar10_subset_metrics.json",
        "loss_curve_path": "cifar10_subset_loss_curve.png",
        "accuracy_curve_path": "cifar10_subset_accuracy_curve.png",
    }
    for key, expected_name in expected_names.items():
        assert first[key].name == expected_name
        assert first[key].is_file()

    metrics = load_metrics(first["metrics_path"])
    assert metrics == first["metrics_history"]
    assert first["metrics_history"] == second["metrics_history"]
    assert first["final_metrics"]["train_total_samples"] == 4
    assert first["final_metrics"]["eval_total_samples"] == 3

    with np.load(first["checkpoint_path"]) as first_checkpoint:
        with np.load(second["checkpoint_path"]) as second_checkpoint:
            assert first_checkpoint.files == second_checkpoint.files
            for key in first_checkpoint.files:
                np.testing.assert_array_equal(
                    first_checkpoint[key],
                    second_checkpoint[key],
                )
