import csv
import json
import math
from pathlib import Path

import pytest

from experiments.fgsm import plot_fgsm_run


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _metric_rows() -> list[dict[str, float | int | str]]:
    return [
        {
            "run_id": "synthetic-run",
            "backend": "cupy",
            "split": "test",
            "seed": 42,
            "batch_size": 32,
            "requested_max_samples": 1000,
            "epsilon": 0.0,
            "total_samples": 1000,
            "clean_correct": 450,
            "adversarial_correct": 450,
            "clean_correct_samples": 450,
            "successful_attacks": 0,
            "clean_accuracy": 0.45,
            "adversarial_accuracy": 0.45,
            "accuracy_drop": 0.0,
            "attack_success_rate": 0.0,
        },
        {
            "run_id": "synthetic-run",
            "backend": "cupy",
            "split": "test",
            "seed": 42,
            "batch_size": 32,
            "requested_max_samples": 1000,
            "epsilon": 1.0 / 255.0,
            "total_samples": 1000,
            "clean_correct": 450,
            "adversarial_correct": 300,
            "clean_correct_samples": 450,
            "successful_attacks": 150,
            "clean_accuracy": 0.45,
            "adversarial_accuracy": 0.30,
            "accuracy_drop": 0.15,
            "attack_success_rate": 1.0 / 3.0,
        },
        {
            "run_id": "synthetic-run",
            "backend": "cupy",
            "split": "test",
            "seed": 42,
            "batch_size": 32,
            "requested_max_samples": 1000,
            "epsilon": 2.0 / 255.0,
            "total_samples": 1000,
            "clean_correct": 450,
            "adversarial_correct": 220,
            "clean_correct_samples": 450,
            "successful_attacks": 230,
            "clean_accuracy": 0.45,
            "adversarial_accuracy": 0.22,
            "accuracy_drop": 0.23,
            "attack_success_rate": 230.0 / 450.0,
        },
    ]


def _write_metrics_csv(run_dir: Path, rows: list[dict[str, object]]) -> None:
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_synthetic_run(
    tmp_path: Path,
    *,
    status: str = "COMPLETED",
    rows: list[dict[str, object]] | None = None,
    checksum_matches: bool = True,
    timing: dict[str, object] | None = None,
) -> Path:
    run_dir = tmp_path / "runs" / "synthetic-run"
    run_dir.mkdir(parents=True)
    metric_rows = rows if rows is not None else _metric_rows()
    epsilons = [row["epsilon"] for row in metric_rows]
    timing_payload = timing or {
        "schema_version": 1,
        "timing_method": "time.perf_counter",
        "gpu_synchronization": (
            "cupy Stream.null synchronized before and after evaluation"
        ),
        "total_wall_seconds": 20.0,
        "evaluation_wall_seconds": 12.5,
        "sample_count": 1000,
        "epsilon_count": len(metric_rows),
        "sample_epsilon_pairs": 1000 * len(metric_rows),
        "evaluation_sample_epsilon_pairs_per_second": (
            1000 * len(metric_rows) / 12.5
        ),
    }

    _write_json(
        run_dir / "config.json",
        {
            "schema_version": 1,
            "experiment_type": "fgsm_epsilon_sweep",
            "backend": "cupy",
            "data_dir": "data/raw",
            "checkpoint_path": "results/checkpoints/portfolio_baseline_best.npz",
            "split": "test",
            "max_samples": 1000,
            "batch_size": 32,
            "epsilon_values": epsilons,
            "seed": 42,
            "output_root": "results/runs",
            "run_id": "synthetic-run",
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
                "branch": "ewp3c",
                "dirty": False,
                "status_short": "",
            },
        },
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "schema_version": 1,
            "metric_fields": list(metric_rows[0].keys()),
            "results": metric_rows,
        },
    )
    _write_metrics_csv(run_dir, metric_rows)
    _write_json(run_dir / "timing.json", timing_payload)
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": 1,
            "run_id": "synthetic-run",
            "status": "COMPLETED",
            "backend": "cupy",
            "split": "test",
            "sample_count": 1000,
            "batch_size": 32,
            "epsilon_values": epsilons,
            "epsilon_count": len(metric_rows),
            "timing": timing_payload,
            "dataset": {
                "archive_checksum_matches": checksum_matches,
                "available_split_samples": 10000,
                "evaluated_samples": 1000,
                "class_count": 10,
            },
        },
    )
    _write_json(
        run_dir / "status.json",
        {
            "schema_version": 1,
            "status": status,
            "started_at": "2026-08-11T00:00:00Z",
            "ended_at": "2026-08-11T00:01:00Z",
        },
    )
    return run_dir


def test_load_and_validate_run_artifacts_preserves_epsilon_order(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_run(tmp_path)

    artifacts = plot_fgsm_run.load_run_artifacts(run_dir)
    rows = plot_fgsm_run.validate_run_artifacts(
        artifacts,
        expected_sample_count=1000,
        expected_epsilons=(0.0, 1.0 / 255.0, 2.0 / 255.0),
    )

    assert [row["epsilon"] for row in rows] == [
        0.0,
        1.0 / 255.0,
        2.0 / 255.0,
    ]
    assert plot_fgsm_run.epsilon_labels(rows) == ["0", "1/255", "2/255"]


def test_curate_fgsm_run_writes_summary_plots_and_metadata(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_run(tmp_path)

    outputs = plot_fgsm_run.curate_fgsm_run(
        run_dir,
        tmp_path / "curated",
        expected_sample_count=1000,
        expected_epsilons=(0.0, 1.0 / 255.0, 2.0 / 255.0),
    )

    expected_files = (
        "robustness_summary",
        "run_metadata",
        "timing_summary",
        "accuracy_vs_epsilon",
        "attack_success_rate_vs_epsilon",
        "accuracy_drop_vs_epsilon",
        "runtime_throughput_summary",
    )
    for key in expected_files:
        assert outputs[key].is_file()

    with outputs["robustness_summary"].open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["epsilon_label"] for row in rows] == ["0", "1/255", "2/255"]
    assert rows[0]["total_samples"] == "1000"

    metadata = json.loads(outputs["run_metadata"].read_text(encoding="utf-8"))
    timing = json.loads(outputs["timing_summary"].read_text(encoding="utf-8"))
    assert metadata["gpu"] == "NVIDIA GeForce RTX 2080 Ti"
    assert metadata["checkpoint_path"] == (
        "results/checkpoints/portfolio_baseline_best.npz"
    )
    assert metadata["git_commit"] == "abc123"
    assert timing["sample_epsilon_pairs"] == 3000
    assert math.isclose(
        timing["evaluation_sample_epsilon_pairs_per_second"],
        240.0,
    )


def test_curate_fgsm_run_rejects_existing_outputs_without_overwrite(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_run(tmp_path)
    output_root = tmp_path / "curated"

    plot_fgsm_run.curate_fgsm_run(run_dir, output_root)

    with pytest.raises(FileExistsError, match="already exists"):
        plot_fgsm_run.curate_fgsm_run(run_dir, output_root)

    outputs = plot_fgsm_run.curate_fgsm_run(
        run_dir,
        output_root,
        overwrite=True,
    )
    assert outputs["runtime_throughput_summary"].is_file()


def test_load_run_artifacts_rejects_missing_required_file(tmp_path: Path) -> None:
    run_dir = _write_synthetic_run(tmp_path)
    (run_dir / "metrics.csv").unlink()

    with pytest.raises(FileNotFoundError, match="metrics.csv"):
        plot_fgsm_run.load_run_artifacts(run_dir)


def test_validate_run_artifacts_rejects_invalid_status(tmp_path: Path) -> None:
    run_dir = _write_synthetic_run(tmp_path, status="FAILED")
    artifacts = plot_fgsm_run.load_run_artifacts(run_dir)

    with pytest.raises(ValueError, match="COMPLETED"):
        plot_fgsm_run.validate_run_artifacts(artifacts)


def test_validate_run_artifacts_rejects_nonfinite_metric(tmp_path: Path) -> None:
    rows = _metric_rows()
    rows[1] = {**rows[1], "adversarial_accuracy": float("nan")}
    run_dir = _write_synthetic_run(tmp_path, rows=rows)
    artifacts = plot_fgsm_run.load_run_artifacts(run_dir)

    with pytest.raises(ValueError, match="adversarial_accuracy"):
        plot_fgsm_run.validate_run_artifacts(artifacts)


def test_validate_run_artifacts_rejects_bad_dataset_checksum(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_run(tmp_path, checksum_matches=False)
    artifacts = plot_fgsm_run.load_run_artifacts(run_dir)

    with pytest.raises(ValueError, match="Dataset checksum"):
        plot_fgsm_run.validate_run_artifacts(artifacts)
