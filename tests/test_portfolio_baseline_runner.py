from pathlib import Path

import numpy as np
import pytest

from experiments.baseline import train_portfolio_baseline as runner


def _synthetic_cifar10_like_data():
    rng = np.random.default_rng(123)
    train_images = rng.random((48, 3, 32, 32), dtype=np.float32)
    train_labels = np.arange(48, dtype=np.int64) % 10
    test_images = rng.random((20, 3, 32, 32), dtype=np.float32)
    test_labels = np.arange(20, dtype=np.int64) % 10
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    return train_images, train_labels, test_images, test_labels, class_names


def test_portfolio_baseline_config_rejects_invalid_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="epochs must be"):
        runner.PortfolioBaselineConfig(
            train_samples=20,
            validation_samples=10,
            test_samples=10,
            epochs=0,
            output_dir=tmp_path,
        )


def test_run_portfolio_baseline_creates_expected_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "load_cifar10",
        lambda data_dir: _synthetic_cifar10_like_data(),
    )
    config = runner.PortfolioBaselineConfig(
        train_samples=20,
        validation_samples=10,
        test_samples=10,
        batch_size=5,
        epochs=2,
        learning_rate=0.01,
        seed=42,
        output_dir=tmp_path / "baseline",
    )

    result = runner.run_portfolio_baseline(config=config, data_dir=tmp_path)

    expected_paths = [
        result["checkpoint_path"],
        result["history_path"],
        result["final_metrics_path"],
        result["training_loss_curve_path"],
        result["train_validation_accuracy_curve_path"],
        result["confusion_matrix_path"],
    ]
    for path in expected_paths:
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0

    assert len(result["metrics_history"]) == config.epochs
    assert result["confusion_matrix"].shape == (10, 10)
    assert result["confusion_matrix"].sum() == config.test_samples
    final_metrics = result["final_metrics"]
    assert final_metrics["train_samples"] == config.train_samples
    assert final_metrics["validation_samples"] == config.validation_samples
    assert final_metrics["test_samples"] == config.test_samples
    assert 0.0 <= final_metrics["final_validation_accuracy"] <= 1.0
    assert 0.0 <= final_metrics["final_test_accuracy"] <= 1.0
