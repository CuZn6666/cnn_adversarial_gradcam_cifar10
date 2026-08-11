import json
import math
import re
from pathlib import Path

import pytest

from experiments.fgsm import run_fgsm_benchmark


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fake_runner(config):
    run_dir = Path(config.output_root) / str(config.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    repeat_match = re.search(r"(?:repeat|warmup)(\d+)$", config.run_id)
    repeat_index = int(repeat_match.group(1)) if repeat_match else 0
    backend_factor = 1.0 if config.backend == "numpy" else 0.2
    evaluation_seconds = (
        config.max_samples
        * len(config.epsilon_values)
        * backend_factor
        / config.batch_size
        / 10.0
        + repeat_index * 0.01
    )
    total_seconds = evaluation_seconds + 0.5
    sample_epsilon_pairs = config.max_samples * len(config.epsilon_values)
    timing = {
        "evaluation_wall_seconds": evaluation_seconds,
        "total_wall_seconds": total_seconds,
        "sample_epsilon_pairs": sample_epsilon_pairs,
        "evaluation_sample_epsilon_pairs_per_second": (
            sample_epsilon_pairs / evaluation_seconds
        ),
    }
    environment_path = run_dir / "environment.json"
    _write_json(
        environment_path,
        {
            "cupy": {"gpu_name": "NVIDIA GeForce RTX 2080 Ti"},
            "git": {"commit": "abc123"},
        },
    )
    return {
        "run_dir": run_dir,
        "timing": timing,
        "artifacts": {"environment": str(environment_path)},
    }


def _small_config(tmp_path: Path, **overrides) -> run_fgsm_benchmark.FGSBenchmarkConfig:
    values = {
        "data_dir": tmp_path / "data",
        "checkpoint_path": tmp_path / "checkpoint.npz",
        "raw_run_output_root": tmp_path / "runs",
        "benchmark_output_root": tmp_path / "benchmarks",
        "benchmark_id": "benchmark-test",
        "sample_counts": (100,),
        "sample_scaling_backends": ("numpy", "cupy"),
        "sample_scaling_batch_size": 32,
        "batch_sizes": (8,),
        "batch_scaling_backends": ("cupy",),
        "batch_scaling_sample_count": 100,
        "epsilon_values": (0.0, 4.0 / 255.0),
        "repeats": 2,
        "warmup_runs": 1,
    }
    values.update(overrides)
    return run_fgsm_benchmark.FGSBenchmarkConfig(**values)


def test_cli_parses_benchmark_config(tmp_path: Path) -> None:
    args = run_fgsm_benchmark.parse_args(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--checkpoint",
            str(tmp_path / "model.npz"),
            "--sample-counts",
            "100,250",
            "--sample-scaling-backends",
            "numpy,cupy",
            "--sample-scaling-batch-size",
            "16",
            "--batch-sizes",
            "8,32",
            "--batch-scaling-backends",
            "numpy,cupy",
            "--batch-scaling-sample-count",
            "500",
            "--epsilons",
            "0,4/255",
            "--repeats",
            "3",
            "--warmup-runs",
            "1",
            "--benchmark-id",
            "manual-benchmark",
        ]
    )

    config = run_fgsm_benchmark.config_from_args(args)

    assert config.data_dir == tmp_path / "data"
    assert config.checkpoint_path == tmp_path / "model.npz"
    assert config.sample_counts == (100, 250)
    assert config.sample_scaling_backends == ("numpy", "cupy")
    assert config.sample_scaling_batch_size == 16
    assert config.batch_sizes == (8, 32)
    assert config.batch_scaling_backends == ("numpy", "cupy")
    assert config.batch_scaling_sample_count == 500
    assert config.epsilon_values == (0.0, 4.0 / 255.0)
    assert config.repeats == 3
    assert config.warmup_runs == 1
    assert config.benchmark_id == "manual-benchmark"


def test_build_benchmark_plan_includes_warmups_and_repeats(tmp_path: Path) -> None:
    config = _small_config(
        tmp_path,
        sample_counts=(100, 200),
        batch_sizes=(8, 16),
        repeats=2,
        warmup_runs=1,
    )

    points = run_fgsm_benchmark.build_benchmark_plan(config)

    assert len(points) == (2 * 2 * 3) + (2 * 3)
    first = points[0]
    assert first.suite == "sample_count_scaling"
    assert first.measurement_type == "warmup"
    assert first.repeat_index == 0
    measured_points = [point for point in points if point.measurement_type == "measured"]
    assert {point.repeat_index for point in measured_points} == {0, 1}


def test_batch_scaling_can_generate_matched_cpu_gpu_workloads(
    tmp_path: Path,
) -> None:
    config = _small_config(
        tmp_path,
        run_sample_scaling=False,
        batch_sizes=(8, 16, 32),
        batch_scaling_backends=("numpy", "cupy"),
        repeats=3,
        warmup_runs=1,
    )

    points = run_fgsm_benchmark.build_benchmark_plan(config)

    assert len(points) == 3 * 2 * 4
    measured = [point for point in points if point.measurement_type == "measured"]
    assert {(point.batch_size, point.backend) for point in measured} == {
        (8, "numpy"),
        (8, "cupy"),
        (16, "numpy"),
        (16, "cupy"),
        (32, "numpy"),
        (32, "cupy"),
    }
    assert {point.repeat_index for point in measured} == {0, 1, 2}


def test_benchmark_run_ids_are_unique_and_valid(tmp_path: Path) -> None:
    config = _small_config(tmp_path, sample_counts=(100, 250), batch_sizes=(8, 16))
    run_ids = [
        run_fgsm_benchmark.benchmark_run_id(config, point)
        for point in run_fgsm_benchmark.build_benchmark_plan(config)
    ]

    assert len(run_ids) == len(set(run_ids))
    assert all(run_fgsm_benchmark.RUN_ID_PATTERN.fullmatch(run_id) for run_id in run_ids)


def test_aggregation_and_speedup_calculation() -> None:
    rows = [
        {
            "benchmark_id": "bench",
            "suite": "sample_count_scaling",
            "backend": "numpy",
            "sample_count": 100,
            "batch_size": 32,
            "epsilon_values": "0,0.01568627450980392",
            "epsilon_labels": "0,4/255",
            "measurement_type": "measured",
            "repeat_index": 0,
            "status": "COMPLETED",
            "evaluation_wall_seconds": 10.0,
            "total_wall_seconds": 11.0,
            "evaluation_sample_epsilon_pairs_per_second": 20.0,
        },
        {
            "benchmark_id": "bench",
            "suite": "sample_count_scaling",
            "backend": "numpy",
            "sample_count": 100,
            "batch_size": 32,
            "epsilon_values": "0,0.01568627450980392",
            "epsilon_labels": "0,4/255",
            "measurement_type": "measured",
            "repeat_index": 1,
            "status": "COMPLETED",
            "evaluation_wall_seconds": 12.0,
            "total_wall_seconds": 13.0,
            "evaluation_sample_epsilon_pairs_per_second": 18.0,
        },
        {
            "benchmark_id": "bench",
            "suite": "sample_count_scaling",
            "backend": "cupy",
            "sample_count": 100,
            "batch_size": 32,
            "epsilon_values": "0,0.01568627450980392",
            "epsilon_labels": "0,4/255",
            "measurement_type": "measured",
            "repeat_index": 0,
            "status": "COMPLETED",
            "evaluation_wall_seconds": 2.0,
            "total_wall_seconds": 3.0,
            "evaluation_sample_epsilon_pairs_per_second": 100.0,
        },
        {
            "benchmark_id": "bench",
            "suite": "sample_count_scaling",
            "backend": "cupy",
            "sample_count": 100,
            "batch_size": 32,
            "epsilon_values": "0,0.01568627450980392",
            "epsilon_labels": "0,4/255",
            "measurement_type": "measured",
            "repeat_index": 1,
            "status": "COMPLETED",
            "evaluation_wall_seconds": 3.0,
            "total_wall_seconds": 4.0,
            "evaluation_sample_epsilon_pairs_per_second": 90.0,
        },
    ]

    summaries = run_fgsm_benchmark.aggregate_benchmark_rows(rows)
    speedups = run_fgsm_benchmark.compute_speedup_rows(rows, summaries)

    numpy_summary = next(row for row in summaries if row["backend"] == "numpy")
    assert numpy_summary["completed_repeats"] == 2
    assert numpy_summary["evaluation_wall_seconds_mean"] == 11.0
    assert math.isclose(
        numpy_summary["evaluation_wall_seconds_std"],
        math.sqrt(2.0),
    )

    assert len(speedups) == 1
    speedup = speedups[0]
    assert speedup["evaluation_speedup_mean"] == 11.0 / 2.5
    assert speedup["evaluation_speedup_median"] == 11.0 / 2.5
    assert speedup["matched_repeats"] == 2
    assert speedup["paired_evaluation_speedup_median"] == 4.5


def test_batch_size_speedup_and_crossover_detection() -> None:
    rows = []
    for batch_size, cpu_time, gpu_time in (
        (8, 8.0, 10.0),
        (16, 8.0, 6.0),
        (32, 8.0, 4.0),
    ):
        for backend, evaluation_time in (
            ("numpy", cpu_time),
            ("cupy", gpu_time),
        ):
            rows.append(
                {
                    "benchmark_id": "bench",
                    "suite": "batch_size_scaling",
                    "backend": backend,
                    "sample_count": 1000,
                    "batch_size": batch_size,
                    "epsilon_values": "0,0.01568627450980392",
                    "epsilon_labels": "0,4/255",
                    "measurement_type": "measured",
                    "repeat_index": 0,
                    "status": "COMPLETED",
                    "evaluation_wall_seconds": evaluation_time,
                    "total_wall_seconds": evaluation_time + 1.0,
                    "evaluation_sample_epsilon_pairs_per_second": (
                        2000.0 / evaluation_time
                    ),
                }
            )

    summaries = run_fgsm_benchmark.aggregate_benchmark_rows(rows)
    speedups = run_fgsm_benchmark.compute_speedup_rows(rows, summaries)
    crossover = run_fgsm_benchmark.detect_speedup_crossover(speedups)

    assert [row["batch_size"] for row in speedups] == [8, 16, 32]
    assert [row["evaluation_speedup_median"] for row in speedups] == [
        0.8,
        8.0 / 6.0,
        2.0,
    ]
    assert crossover["first_gpu_faster_batch_size"] == 16
    assert crossover["max_speedup_batch_size"] == 32
    assert crossover["max_speedup"] == 2.0


def test_run_fgsm_benchmark_writes_artifacts_with_fake_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config(tmp_path, batch_scaling_backends=("numpy", "cupy"))
    monkeypatch.setattr(run_fgsm_benchmark, "run_fgsm_experiment", _fake_runner)

    result = run_fgsm_benchmark.run_fgsm_benchmark(config)

    benchmark_dir = tmp_path / "benchmarks" / "benchmark-test"
    assert result["benchmark_dir"] == benchmark_dir
    for filename in (
        "config.json",
        "benchmark_runs.csv",
        "benchmark_runs.json",
        "benchmark_summary.csv",
        "benchmark_summary.json",
        "speedup_summary.csv",
        "speedup_summary.json",
        "status.json",
    ):
        assert (benchmark_dir / filename).is_file()

    status = json.loads((benchmark_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETED"
    assert len(result["rows"]) == 12
    assert all(row["status"] == "COMPLETED" for row in result["rows"])
    assert (benchmark_dir / "plots" / "runtime_vs_sample_count.png").is_file()
    assert (benchmark_dir / "plots" / "speedup_vs_batch_size.png").is_file()
    assert (benchmark_dir / "plots" / "cupy_throughput_vs_batch_size.png").is_file()
    crossover = json.loads(
        (benchmark_dir / "crossover_analysis.json").read_text(encoding="utf-8")
    )
    assert crossover["tested_batch_sizes"] == [8]


def test_run_fgsm_benchmark_preserves_partial_failure_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config(
        tmp_path,
        run_sample_scaling=False,
        batch_sizes=(8, 16),
        batch_scaling_backends=("cupy",),
        repeats=1,
        warmup_runs=0,
    )

    def fake_runner_with_failure(config):
        if config.batch_size == 16:
            raise RuntimeError("synthetic batch-size failure")
        return _fake_runner(config)

    monkeypatch.setattr(
        run_fgsm_benchmark,
        "run_fgsm_experiment",
        fake_runner_with_failure,
    )

    result = run_fgsm_benchmark.run_fgsm_benchmark(config)

    statuses = [row["status"] for row in result["rows"]]
    assert statuses == ["COMPLETED", "FAILED"]
    status = json.loads(
        (result["benchmark_dir"] / "status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "COMPLETED_WITH_FAILURES"
    summary_rows = json.loads(
        (result["benchmark_dir"] / "benchmark_summary.json").read_text(
            encoding="utf-8"
        )
    )["rows"]
    assert any(row["completed_repeats"] == 1 for row in summary_rows)
    assert any(row["failed_repeats"] == 1 for row in summary_rows)


def test_plot_benchmark_results_from_synthetic_summary(tmp_path: Path) -> None:
    summaries = [
        {
            "benchmark_id": "bench",
            "suite": "sample_count_scaling",
            "backend": "numpy",
            "sample_count": 100,
            "batch_size": 32,
            "epsilon_values": "0,0.01568627450980392",
            "epsilon_labels": "0,4/255",
            "completed_repeats": 3,
            "evaluation_wall_seconds_median": 10.0,
            "evaluation_wall_seconds_std": 1.0,
            "throughput_median": 20.0,
            "throughput_std": 2.0,
        },
        {
            "benchmark_id": "bench",
            "suite": "sample_count_scaling",
            "backend": "cupy",
            "sample_count": 100,
            "batch_size": 32,
            "epsilon_values": "0,0.01568627450980392",
            "epsilon_labels": "0,4/255",
            "completed_repeats": 3,
            "evaluation_wall_seconds_median": 2.0,
            "evaluation_wall_seconds_std": 0.2,
            "throughput_median": 100.0,
            "throughput_std": 5.0,
        },
        {
            "benchmark_id": "bench",
            "suite": "batch_size_scaling",
            "backend": "numpy",
            "sample_count": 1000,
            "batch_size": 8,
            "epsilon_values": "0,0.01568627450980392",
            "epsilon_labels": "0,4/255",
            "completed_repeats": 3,
            "evaluation_wall_seconds_median": 5.0,
            "evaluation_wall_seconds_std": 0.5,
            "throughput_median": 200.0,
            "throughput_std": 8.0,
        },
        {
            "benchmark_id": "bench",
            "suite": "batch_size_scaling",
            "backend": "cupy",
            "sample_count": 1000,
            "batch_size": 8,
            "epsilon_values": "0,0.01568627450980392",
            "epsilon_labels": "0,4/255",
            "completed_repeats": 3,
            "evaluation_wall_seconds_median": 4.0,
            "evaluation_wall_seconds_std": 0.4,
            "throughput_median": 250.0,
            "throughput_std": 10.0,
        },
        {
            "benchmark_id": "bench",
            "suite": "batch_size_scaling",
            "backend": "numpy",
            "sample_count": 1000,
            "batch_size": 16,
            "epsilon_values": "0,0.01568627450980392",
            "epsilon_labels": "0,4/255",
            "completed_repeats": 3,
            "evaluation_wall_seconds_median": 5.0,
            "evaluation_wall_seconds_std": 0.5,
            "throughput_median": 200.0,
            "throughput_std": 8.0,
        },
        {
            "benchmark_id": "bench",
            "suite": "batch_size_scaling",
            "backend": "cupy",
            "sample_count": 1000,
            "batch_size": 16,
            "epsilon_values": "0,0.01568627450980392",
            "epsilon_labels": "0,4/255",
            "completed_repeats": 3,
            "evaluation_wall_seconds_median": 3.0,
            "evaluation_wall_seconds_std": 0.3,
            "throughput_median": 333.0,
            "throughput_std": 11.0,
        },
    ]
    speedups = [
        {
            "suite": "sample_count_scaling",
            "sample_count": 100,
            "evaluation_speedup_median": 5.0,
        },
        {
            "suite": "batch_size_scaling",
            "batch_size": 8,
            "evaluation_speedup_median": 1.25,
        },
        {
            "suite": "batch_size_scaling",
            "batch_size": 16,
            "evaluation_speedup_median": 5.0 / 3.0,
        },
    ]

    outputs = run_fgsm_benchmark.plot_benchmark_results(
        tmp_path,
        summaries,
        speedups,
    )

    assert set(outputs) == {
        "runtime_vs_sample_count",
        "throughput_vs_sample_count",
        "speedup_vs_sample_count",
        "runtime_vs_batch_size",
        "throughput_vs_batch_size",
        "speedup_vs_batch_size",
        "cupy_runtime_vs_batch_size",
        "cupy_throughput_vs_batch_size",
    }
    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_prepare_benchmark_directory_rejects_collision(tmp_path: Path) -> None:
    config = _small_config(tmp_path)

    first = run_fgsm_benchmark.prepare_benchmark_directory(config)

    assert first == tmp_path / "benchmarks" / "benchmark-test"
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        run_fgsm_benchmark.prepare_benchmark_directory(config)

    overwrite_config = _small_config(tmp_path, overwrite=True)
    assert run_fgsm_benchmark.prepare_benchmark_directory(overwrite_config) == first
