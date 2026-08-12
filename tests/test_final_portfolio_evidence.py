import csv
import json
from pathlib import Path

import pytest

from experiments import generate_final_portfolio_evidence as portfolio


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_synthetic_ewp3d(path: Path) -> None:
    benchmark_rows = []
    speedup_rows = []
    for batch_size, numpy_tput, cupy_tput, speedup in (
        (8, 490.0, 120.0, 0.25),
        (16, 575.0, 240.0, 0.42),
        (32, 622.0, 475.0, 0.76),
        (64, 644.0, 945.0, 1.47),
        (128, 645.0, 1856.0, 2.88),
    ):
        for backend, throughput in (("numpy", numpy_tput), ("cupy", cupy_tput)):
            benchmark_rows.append(
                {
                    "benchmark_id": "synthetic-benchmark",
                    "suite": "batch_size_scaling",
                    "backend": backend,
                    "sample_count": 1000,
                    "batch_size": batch_size,
                    "epsilon_values": "0,0.015686274509803921",
                    "epsilon_labels": "0,4/255",
                    "completed_repeats": 3,
                    "failed_repeats": 0,
                    "evaluation_wall_seconds_mean": 1.0,
                    "evaluation_wall_seconds_median": 1.0,
                    "evaluation_wall_seconds_std": 0.01,
                    "evaluation_wall_seconds_min": 0.99,
                    "evaluation_wall_seconds_max": 1.01,
                    "total_wall_seconds_mean": 1.5,
                    "total_wall_seconds_median": 1.5,
                    "total_wall_seconds_std": 0.01,
                    "total_wall_seconds_min": 1.49,
                    "total_wall_seconds_max": 1.51,
                    "throughput_mean": throughput,
                    "throughput_median": throughput,
                    "throughput_std": 1.0,
                    "throughput_min": throughput - 1,
                    "throughput_max": throughput + 1,
                }
            )
        speedup_rows.append(
            {
                "benchmark_id": "synthetic-benchmark",
                "suite": "batch_size_scaling",
                "sample_count": 1000,
                "batch_size": batch_size,
                "epsilon_values": "0,0.015686274509803921",
                "epsilon_labels": "0,4/255",
                "matched_repeats": 3,
                "cpu_backend": "numpy",
                "gpu_backend": "cupy",
                "cpu_evaluation_wall_seconds_mean": 1.0,
                "gpu_evaluation_wall_seconds_mean": 1.0 / speedup,
                "evaluation_speedup_mean": speedup,
                "cpu_evaluation_wall_seconds_median": 1.0,
                "gpu_evaluation_wall_seconds_median": 1.0 / speedup,
                "evaluation_speedup_median": speedup,
                "paired_evaluation_speedup_mean": speedup,
                "paired_evaluation_speedup_median": speedup,
                "paired_evaluation_speedup_std": 0.01,
                "paired_evaluation_speedup_min": speedup - 0.01,
                "paired_evaluation_speedup_max": speedup + 0.01,
                "total_wall_speedup_mean": speedup,
                "total_wall_speedup_median": speedup,
            }
        )
    _write_csv(path / "benchmark_summary.csv", benchmark_rows)
    _write_csv(path / "speedup_summary.csv", speedup_rows)
    _write_json(
        path / "crossover_analysis.json",
        {
            "schema_version": 1,
            "suite": "batch_size_scaling",
            "speedup_metric": "evaluation_speedup_median",
            "break_even_speedup": 1.0,
            "tested_batch_sizes": [8, 16, 32, 64, 128],
            "first_gpu_faster_batch_size": 64,
            "first_gpu_faster_speedup": 1.47,
            "max_speedup_batch_size": 128,
            "max_speedup": 2.88,
        },
    )


def _write_synthetic_ewp3e(path: Path, *, backend: str = "cupy") -> None:
    rows = []
    for label, epsilon, adversarial_accuracy, attack_success_rate in (
        ("0", 0.0, 0.4639, 0.0),
        ("1/255", 1 / 255, 0.3020, 0.3490),
        ("2/255", 2 / 255, 0.1854, 0.6003),
        ("4/255", 4 / 255, 0.0743, 0.8398),
        ("8/255", 8 / 255, 0.0099, 0.9787),
        ("12/255", 12 / 255, 0.0017, 0.9963),
        ("16/255", 16 / 255, 0.0004, 0.9991),
    ):
        rows.append(
            {
                "epsilon": epsilon,
                "epsilon_label": label,
                "total_samples": 10000,
                "clean_correct": 4639,
                "adversarial_correct": int(round(adversarial_accuracy * 10000)),
                "clean_correct_samples": 4639,
                "successful_attacks": int(round(attack_success_rate * 4639)),
                "clean_accuracy": 0.4639,
                "adversarial_accuracy": adversarial_accuracy,
                "accuracy_drop": 0.4639 - adversarial_accuracy,
                "attack_success_rate": attack_success_rate,
            }
        )
    rows[0]["adversarial_correct"] = 4639
    rows[0]["successful_attacks"] = 0
    _write_csv(path / "robustness_summary.csv", rows)
    _write_json(
        path / "timing_summary.json",
        {
            "schema_version": 1,
            "run_id": "synthetic-run",
            "backend": backend,
            "sample_count": 10000,
            "epsilon_count": 7,
            "sample_epsilon_pairs": 70000,
            "total_wall_seconds": 38.0,
            "evaluation_wall_seconds": 37.0,
            "evaluation_sample_epsilon_pairs_per_second": 1891.39,
            "timing_method": "time.perf_counter",
            "gpu_synchronization": "cupy Stream.null synchronized before and after evaluation",
        },
    )
    _write_json(
        path / "run_metadata.json",
        {
            "schema_version": 1,
            "run_id": "synthetic-run",
            "backend": backend,
            "gpu": "NVIDIA GeForce RTX 2080 Ti",
            "cupy_version": "14.1.1",
            "numpy_version": "2.4.6",
            "python_version": "3.12.13",
            "dataset_checksum_matches": True,
            "sample_count": 10000,
            "batch_size": 128,
            "seed": 42,
        },
    )


def test_build_portfolio_summary_uses_curated_sources(tmp_path: Path) -> None:
    ewp3d_dir = tmp_path / "ewp3d"
    ewp3e_dir = tmp_path / "ewp3e"
    _write_synthetic_ewp3d(ewp3d_dir)
    _write_synthetic_ewp3e(ewp3e_dir)

    summary = portfolio.build_portfolio_summary(
        ewp3d_dir=ewp3d_dir,
        ewp3e_dir=ewp3e_dir,
    )

    assert summary["full_test_sample_count"] == 10000
    assert summary["backend"] == "cupy"
    assert summary["gpu"] == "NVIDIA GeForce RTX 2080 Ti"
    assert summary["clean_accuracy"] == 0.4639
    assert summary["adversarial_accuracy_4_255"] == 0.0743
    assert summary["adversarial_accuracy_16_255"] == 0.0004
    assert summary["first_tested_gpu_faster_batch_size"] == 64
    assert summary["best_tested_speedup"] == 2.88
    assert summary["best_tested_batch_size"] == 128
    assert summary["full_10k_evaluation_throughput"] == 1891.39


def test_build_portfolio_evidence_writes_summary_and_plots(tmp_path: Path) -> None:
    ewp3d_dir = tmp_path / "ewp3d"
    ewp3e_dir = tmp_path / "ewp3e"
    output_dir = tmp_path / "portfolio"
    _write_synthetic_ewp3d(ewp3d_dir)
    _write_synthetic_ewp3e(ewp3e_dir)

    outputs = portfolio.build_portfolio_evidence(
        ewp3d_dir=ewp3d_dir,
        ewp3e_dir=ewp3e_dir,
        output_dir=output_dir,
    )

    assert outputs["portfolio_summary_csv"].is_file()
    assert outputs["portfolio_summary_json"].is_file()
    for key in ("final_performance_summary", "final_robustness_summary"):
        data = outputs[key].read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 1000

    rows = list(csv.DictReader(outputs["portfolio_summary_csv"].open()))
    assert len(rows) == 1
    assert rows[0]["best_tested_batch_size"] == "128"


def test_build_portfolio_summary_rejects_unexpected_backend(tmp_path: Path) -> None:
    ewp3d_dir = tmp_path / "ewp3d"
    ewp3e_dir = tmp_path / "ewp3e"
    _write_synthetic_ewp3d(ewp3d_dir)
    _write_synthetic_ewp3e(ewp3e_dir, backend="numpy")

    with pytest.raises(ValueError, match="backend=cupy"):
        portfolio.build_portfolio_summary(
            ewp3d_dir=ewp3d_dir,
            ewp3e_dir=ewp3e_dir,
        )


def test_build_portfolio_evidence_requires_overwrite(tmp_path: Path) -> None:
    ewp3d_dir = tmp_path / "ewp3d"
    ewp3e_dir = tmp_path / "ewp3e"
    output_dir = tmp_path / "portfolio"
    _write_synthetic_ewp3d(ewp3d_dir)
    _write_synthetic_ewp3e(ewp3e_dir)

    portfolio.build_portfolio_evidence(
        ewp3d_dir=ewp3d_dir,
        ewp3e_dir=ewp3e_dir,
        output_dir=output_dir,
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        portfolio.build_portfolio_evidence(
            ewp3d_dir=ewp3d_dir,
            ewp3e_dir=ewp3e_dir,
            output_dir=output_dir,
        )
