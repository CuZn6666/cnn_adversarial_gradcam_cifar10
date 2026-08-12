import csv
import json
from pathlib import Path

import numpy as np
import pytest

from configs.default_config import (
    CIFAR10_ARCHIVE_NAME,
    CIFAR10_EXTRACTED_DIR,
    CIFAR10_MD5,
)
from experiments.fgsm import run_fgsm_experiment
from experiments.pgd import run_pgd_experiment
from src.checkpointing import save_checkpoint
from src.metrics import load_metrics
from src.models import CompactCNN


def _write_staged_data_markers(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / CIFAR10_ARCHIVE_NAME).write_bytes(b"synthetic archive")
    (data_dir / CIFAR10_EXTRACTED_DIR).mkdir()


def _write_checkpoint(path: Path) -> None:
    save_checkpoint(CompactCNN(seed=7), path)


def _synthetic_cifar10():
    rng = np.random.default_rng(123)
    train_images = rng.random((4, 3, 32, 32), dtype=np.float32)
    train_labels = np.array([0, 1, 2, 3], dtype=np.int64)
    test_images = rng.random((3, 3, 32, 32), dtype=np.float32)
    test_labels = np.array([1, 2, 3], dtype=np.int64)
    class_names = [f"class_{index}" for index in range(10)]
    return train_images, train_labels, test_images, test_labels, class_names


def _config(
    tmp_path: Path,
    *,
    run_id: str = "tiny-pgd-run",
    checkpoint_path: Path | None = None,
) -> run_pgd_experiment.PGDExperimentConfig:
    data_dir = tmp_path / "data"
    _write_staged_data_markers(data_dir)
    checkpoint = checkpoint_path or tmp_path / "model.npz"
    if checkpoint_path is None:
        _write_checkpoint(checkpoint)

    return run_pgd_experiment.PGDExperimentConfig(
        backend="numpy",
        data_dir=data_dir,
        checkpoint_path=checkpoint,
        split="test",
        max_samples=2,
        batch_size=1,
        epsilon=1.0 / 255.0,
        alpha=1.0 / 255.0,
        steps=1,
        random_start=False,
        seed=11,
        output_root=tmp_path / "runs",
        run_id=run_id,
    )


def test_cli_parses_effective_pgd_config(tmp_path: Path) -> None:
    args = run_pgd_experiment.parse_args(
        [
            "--backend",
            "numpy",
            "--data-dir",
            str(tmp_path / "data"),
            "--checkpoint",
            str(tmp_path / "checkpoint.npz"),
            "--split",
            "test",
            "--max-samples",
            "32",
            "--batch-size",
            "8",
            "--epsilon",
            "8/255",
            "--alpha",
            "2/255",
            "--steps",
            "10",
            "--random-start",
            "--seed",
            "42",
            "--output-root",
            str(tmp_path / "runs"),
            "--run-id",
            "manual-pgd-run",
        ]
    )

    config = run_pgd_experiment.config_from_args(args)

    assert config.backend == "numpy"
    assert config.data_dir == tmp_path / "data"
    assert config.checkpoint_path == tmp_path / "checkpoint.npz"
    assert config.split == "test"
    assert config.max_samples == 32
    assert config.batch_size == 8
    assert config.epsilon == 8.0 / 255.0
    assert config.alpha == 2.0 / 255.0
    assert config.steps == 10
    assert config.random_start is True
    assert config.seed == 42
    assert config.output_root == tmp_path / "runs"
    assert config.run_id == "manual-pgd-run"


def test_cli_supports_disabling_random_start(tmp_path: Path) -> None:
    args = run_pgd_experiment.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.npz"),
            "--epsilon",
            "4/255",
            "--alpha",
            "1/255",
            "--steps",
            "4",
            "--no-random-start",
        ]
    )

    config = run_pgd_experiment.config_from_args(args)

    assert config.random_start is False
    assert config.epsilon == 4.0 / 255.0
    assert config.alpha == 1.0 / 255.0


def test_config_rejects_invalid_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend"):
        run_pgd_experiment.PGDExperimentConfig(backend="torch")
    with pytest.raises(ValueError, match="batch_size"):
        run_pgd_experiment.PGDExperimentConfig(batch_size=0)
    with pytest.raises(ValueError, match="epsilon"):
        run_pgd_experiment.PGDExperimentConfig(epsilon=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        run_pgd_experiment.PGDExperimentConfig(alpha=0.0, steps=1)
    with pytest.raises(ValueError, match="steps"):
        run_pgd_experiment.PGDExperimentConfig(steps=-1)
    with pytest.raises(ValueError, match="random_start"):
        run_pgd_experiment.PGDExperimentConfig(random_start="yes")
    with pytest.raises(ValueError, match="run_id"):
        run_pgd_experiment.PGDExperimentConfig(
            output_root=tmp_path,
            run_id="../bad",
        )
    with pytest.raises(ValueError, match="exactly one"):
        run_pgd_experiment.parse_single_epsilon("1/255,2/255", name="epsilon")


def test_steps_zero_allows_alpha_zero() -> None:
    config = run_pgd_experiment.PGDExperimentConfig(alpha=0.0, steps=0)

    assert config.alpha == 0.0
    assert config.steps == 0


def test_prepare_run_directory_rejects_collision(tmp_path: Path) -> None:
    config = run_pgd_experiment.PGDExperimentConfig(
        output_root=tmp_path,
        run_id="collision",
    )

    first_run_dir = run_pgd_experiment.prepare_run_directory(config)

    assert first_run_dir == tmp_path / "collision"
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        run_pgd_experiment.prepare_run_directory(config)


def test_write_metrics_artifacts_preserves_pgd_fields(tmp_path: Path) -> None:
    config = run_pgd_experiment.PGDExperimentConfig(
        output_root=tmp_path,
        run_id="metrics",
        max_samples=2,
        batch_size=1,
        epsilon=1.0 / 255.0,
        alpha=1.0 / 255.0,
        steps=1,
        random_start=False,
    )
    run_dir = tmp_path / "metrics"
    run_dir.mkdir()
    pgd_result = {
        "epsilon": 1.0 / 255.0,
        "alpha": 1.0 / 255.0,
        "steps": 1,
        "random_start": False,
        "total_samples": 2,
        "clean_correct": 1,
        "adversarial_correct": 1,
        "clean_correct_samples": 1,
        "successful_attacks": 0,
        "clean_accuracy": 0.5,
        "adversarial_accuracy": 0.5,
        "accuracy_drop": 0.0,
        "attack_success_rate": 0.0,
    }

    artifacts = run_pgd_experiment.write_metrics_artifacts(
        run_dir,
        config,
        pgd_result,
    )

    metrics_json = load_metrics(artifacts["metrics_json"])
    assert metrics_json["attack"] == "pgd_linf"
    assert metrics_json["results"][0]["attack"] == "pgd_linf"
    assert metrics_json["results"][0]["steps"] == 1

    with Path(artifacts["metrics_csv"]).open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["epsilon"] == str(1.0 / 255.0)
    assert rows[0]["random_start"] == "False"


def test_collect_environment_metadata_numpy_does_not_require_cupy() -> None:
    metadata = run_pgd_experiment.collect_environment_metadata("numpy")

    assert metadata["backend_requested"] == "numpy"
    assert metadata["python_version"]
    assert metadata["numpy_version"]
    assert "cupy" in metadata
    assert "git" in metadata


def test_run_pgd_experiment_writes_tiny_numpy_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        run_fgsm_experiment,
        "compute_md5",
        lambda path: CIFAR10_MD5,
    )
    monkeypatch.setattr(
        run_fgsm_experiment,
        "load_cifar10",
        lambda data_dir: _synthetic_cifar10(),
    )

    result = run_pgd_experiment.run_pgd_experiment(config)

    run_dir = tmp_path / "runs" / "tiny-pgd-run"
    assert result["run_dir"] == run_dir
    for filename in (
        "config.json",
        "environment.json",
        "metrics.csv",
        "metrics.json",
        "timing.json",
        "summary.json",
        "status.json",
    ):
        assert (run_dir / filename).is_file()

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))

    assert status["status"] == "COMPLETED"
    assert summary["attack"] == "pgd_linf"
    assert summary["backend"] == "numpy"
    assert summary["dataset"]["evaluated_samples"] == 2
    assert summary["dataset"]["class_count"] == 10
    assert metrics["results"][0]["total_samples"] == 2
    assert metrics["results"][0]["epsilon"] == 1.0 / 255.0
    assert metrics["results"][0]["steps"] == 1
    assert timing["sample_count"] == 2
    assert timing["pgd_steps"] == 1
    assert timing["sample_steps"] == 2
    assert timing["samples_per_second"] > 0.0


def test_run_pgd_experiment_records_failed_status_for_missing_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_checkpoint = tmp_path / "missing.npz"
    config = _config(
        tmp_path,
        run_id="missing-checkpoint",
        checkpoint_path=missing_checkpoint,
    )
    monkeypatch.setattr(
        run_fgsm_experiment,
        "compute_md5",
        lambda path: CIFAR10_MD5,
    )

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run_pgd_experiment.run_pgd_experiment(config)

    status_path = tmp_path / "runs" / "missing-checkpoint" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "FAILED"
    assert status["error"]["type"] == "FileNotFoundError"
