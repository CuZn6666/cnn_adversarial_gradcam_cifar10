import csv
import json
import math
from pathlib import Path

import pytest

from experiments.pgd import plot_pgd_run


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _metric_row() -> dict[str, float | int | str | bool]:
    return {
        "run_id": "synthetic-pgd-run",
        "attack": "pgd_linf",
        "backend": "cupy",
        "split": "test",
        "seed": 42,
        "batch_size": 8,
        "requested_max_samples": 32,
        "epsilon": 8.0 / 255.0,
        "alpha": 2.0 / 255.0,
        "steps": 10,
        "random_start": True,
        "total_samples": 32,
        "clean_correct": 16,
        "adversarial_correct": 4,
        "clean_correct_samples": 16,
        "successful_attacks": 12,
        "clean_accuracy": 0.5,
        "adversarial_accuracy": 0.125,
        "accuracy_drop": 0.375,
        "attack_success_rate": 0.75,
    }


def _write_metrics_csv(run_dir: Path, row: dict[str, object]) -> None:
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


def _write_synthetic_pgd_run(
    tmp_path: Path,
    *,
    status: str = "COMPLETED",
    row: dict[str, object] | None = None,
    checksum_matches: bool = True,
    timing: dict[str, object] | None = None,
) -> Path:
    run_dir = tmp_path / "runs" / "synthetic-pgd-run"
    run_dir.mkdir(parents=True)
    metric_row = row if row is not None else _metric_row()
    timing_payload = timing or {
        "schema_version": 1,
        "timing_method": "time.perf_counter",
        "gpu_synchronization": (
            "cupy Stream.null synchronized before and after evaluation"
        ),
        "total_wall_seconds": 4.0,
        "evaluation_wall_seconds": 3.2,
        "sample_count": 32,
        "pgd_steps": 10,
        "gradient_evaluations": 320,
        "sample_steps": 320,
        "samples_per_second": 10.0,
        "sample_steps_per_second": 100.0,
    }

    _write_json(
        run_dir / "config.json",
        {
            "schema_version": 1,
            "experiment_type": "pgd_linf",
            "attack": "pgd_linf",
            "backend": "cupy",
            "data_dir": "data/raw",
            "checkpoint_path": "results/checkpoints/portfolio_baseline_best.npz",
            "split": "test",
            "max_samples": 32,
            "batch_size": 8,
            "epsilon": 8.0 / 255.0,
            "alpha": 2.0 / 255.0,
            "steps": 10,
            "random_start": True,
            "seed": 42,
            "random_start_seed_strategy": (
                "seed + batch_index for each evaluated batch"
            ),
            "output_root": "results/runs",
            "run_id": "synthetic-pgd-run",
        },
    )
    _write_json(
        run_dir / "environment.json",
        {
            "schema_version": 1,
            "backend_requested": "cupy",
            "python_version": "3.12.13",
            "numpy_version": "2.4.6",
            "hostname": "csg-brook01",
            "cupy": {
                "version": "14.1.1",
                "cuda_runtime_version": 12050,
                "device_count": 1,
                "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
            },
            "git": {
                "commit": "abc123",
                "branch": "ewp4c",
                "dirty": False,
                "status_short": "",
            },
        },
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "schema_version": 1,
            "attack": "pgd_linf",
            "metric_fields": list(metric_row.keys()),
            "results": [metric_row],
        },
    )
    _write_metrics_csv(run_dir, metric_row)
    _write_json(run_dir / "timing.json", timing_payload)
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": 1,
            "run_id": "synthetic-pgd-run",
            "status": "COMPLETED",
            "experiment_type": "pgd_linf",
            "attack": "pgd_linf",
            "backend": "cupy",
            "split": "test",
            "sample_count": 32,
            "batch_size": 8,
            "epsilon": 8.0 / 255.0,
            "alpha": 2.0 / 255.0,
            "steps": 10,
            "random_start": True,
            "timing": timing_payload,
            "dataset": {
                "archive_checksum_matches": checksum_matches,
                "available_split_samples": 10000,
                "evaluated_samples": 32,
                "class_count": 10,
            },
        },
    )
    _write_json(
        run_dir / "status.json",
        {
            "schema_version": 1,
            "status": status,
            "started_at": "2026-08-12T00:00:00Z",
            "ended_at": "2026-08-12T00:01:00Z",
        },
    )
    return run_dir


def test_load_and_validate_pgd_run_artifacts(tmp_path: Path) -> None:
    run_dir = _write_synthetic_pgd_run(tmp_path)

    artifacts = plot_pgd_run.load_run_artifacts(run_dir)
    row = plot_pgd_run.validate_run_artifacts(
        artifacts,
        expected_sample_count=32,
        expected_epsilon=8.0 / 255.0,
        expected_backend="cupy",
        expected_gpu_name="NVIDIA GeForce RTX 2080 Ti",
    )

    assert row["attack"] == "pgd_linf"
    assert row["epsilon"] == 8.0 / 255.0
    assert row["alpha"] == 2.0 / 255.0
    assert row["steps"] == 10
    assert row["random_start"] is True


def test_curate_pgd_run_writes_summary_plot_and_metadata(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_pgd_run(tmp_path)

    outputs = plot_pgd_run.curate_pgd_run(
        run_dir,
        tmp_path / "curated",
        expected_sample_count=32,
        expected_epsilon=8.0 / 255.0,
        expected_backend="cupy",
    )

    for key in (
        "robustness_summary",
        "run_metadata",
        "timing_summary",
        "pgd_smoke_summary",
    ):
        assert outputs[key].is_file()
        assert outputs[key].stat().st_size > 0

    with outputs["robustness_summary"].open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["total_samples"] == "32"
    assert rows[0]["steps"] == "10"
    assert rows[0]["random_start"] == "True"

    metadata = json.loads(outputs["run_metadata"].read_text(encoding="utf-8"))
    timing = json.loads(outputs["timing_summary"].read_text(encoding="utf-8"))
    assert metadata["attack"] == "pgd_linf"
    assert metadata["gpu"] == "NVIDIA GeForce RTX 2080 Ti"
    assert metadata["checkpoint_path"] == (
        "results/checkpoints/portfolio_baseline_best.npz"
    )
    assert metadata["git_commit"] == "abc123"
    assert "Small PGD cluster smoke run" in metadata["interpretation"]
    assert timing["sample_steps"] == 320
    assert math.isclose(timing["sample_steps_per_second"], 100.0)


def test_curate_pgd_run_rejects_existing_outputs_without_overwrite(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_pgd_run(tmp_path)
    output_root = tmp_path / "curated"

    plot_pgd_run.curate_pgd_run(run_dir, output_root)

    with pytest.raises(FileExistsError, match="already exists"):
        plot_pgd_run.curate_pgd_run(run_dir, output_root)

    outputs = plot_pgd_run.curate_pgd_run(
        run_dir,
        output_root,
        overwrite=True,
    )
    assert outputs["pgd_smoke_summary"].is_file()


def test_load_pgd_run_artifacts_rejects_missing_required_file(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_pgd_run(tmp_path)
    (run_dir / "metrics.csv").unlink()

    with pytest.raises(FileNotFoundError, match="metrics.csv"):
        plot_pgd_run.load_run_artifacts(run_dir)


def test_validate_pgd_run_artifacts_rejects_invalid_status(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_pgd_run(tmp_path, status="FAILED")
    artifacts = plot_pgd_run.load_run_artifacts(run_dir)

    with pytest.raises(ValueError, match="COMPLETED"):
        plot_pgd_run.validate_run_artifacts(artifacts)


def test_validate_pgd_run_artifacts_rejects_nonfinite_metric(
    tmp_path: Path,
) -> None:
    row = {**_metric_row(), "adversarial_accuracy": float("nan")}
    run_dir = _write_synthetic_pgd_run(tmp_path, row=row)
    artifacts = plot_pgd_run.load_run_artifacts(run_dir)

    with pytest.raises(ValueError, match="adversarial_accuracy"):
        plot_pgd_run.validate_run_artifacts(artifacts)


def test_validate_pgd_run_artifacts_rejects_out_of_range_metric(
    tmp_path: Path,
) -> None:
    row = {**_metric_row(), "attack_success_rate": 1.2}
    run_dir = _write_synthetic_pgd_run(tmp_path, row=row)
    artifacts = plot_pgd_run.load_run_artifacts(run_dir)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        plot_pgd_run.validate_run_artifacts(artifacts)


def test_validate_pgd_run_artifacts_rejects_mismatched_config(
    tmp_path: Path,
) -> None:
    row = {**_metric_row(), "steps": 9}
    run_dir = _write_synthetic_pgd_run(tmp_path, row=row)
    artifacts = plot_pgd_run.load_run_artifacts(run_dir)

    with pytest.raises(ValueError, match="steps"):
        plot_pgd_run.validate_run_artifacts(artifacts)


def test_validate_pgd_run_artifacts_rejects_unexpected_backend_or_gpu(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_pgd_run(tmp_path)
    artifacts = plot_pgd_run.load_run_artifacts(run_dir)

    with pytest.raises(ValueError, match="Expected backend numpy"):
        plot_pgd_run.validate_run_artifacts(artifacts, expected_backend="numpy")
    with pytest.raises(ValueError, match="Expected GPU"):
        plot_pgd_run.validate_run_artifacts(
            artifacts,
            expected_gpu_name="Different GPU",
        )


def test_validate_pgd_run_artifacts_rejects_bad_dataset_checksum(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_pgd_run(tmp_path, checksum_matches=False)
    artifacts = plot_pgd_run.load_run_artifacts(run_dir)

    with pytest.raises(ValueError, match="Dataset checksum"):
        plot_pgd_run.validate_run_artifacts(artifacts)
