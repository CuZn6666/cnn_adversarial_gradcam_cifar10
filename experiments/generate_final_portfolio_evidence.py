"""Build final portfolio evidence from curated EWP3-D/EWP3-E artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.default_config import PROJECT_ROOT
from src.metrics import save_metrics


DEFAULT_EWP3D_DIR = (
    PROJECT_ROOT
    / "results"
    / "curated"
    / "ewp3d"
    / "20260811T185420645969Z_fgsm_benchmark"
)
DEFAULT_EWP3E_DIR = (
    PROJECT_ROOT
    / "results"
    / "curated"
    / "ewp3e"
    / "20260812T115232600695Z_fgsm_cupy"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "curated" / "portfolio"

SUMMARY_FIELDS = (
    "full_test_sample_count",
    "backend",
    "gpu",
    "cupy_version",
    "numpy_version",
    "python_version",
    "clean_accuracy",
    "adversarial_accuracy_4_255",
    "adversarial_accuracy_8_255",
    "adversarial_accuracy_16_255",
    "attack_success_rate_4_255",
    "attack_success_rate_8_255",
    "attack_success_rate_16_255",
    "first_tested_gpu_faster_batch_size",
    "best_tested_speedup",
    "best_tested_batch_size",
    "full_10k_evaluation_throughput",
    "full_10k_evaluation_wall_seconds",
    "sample_epsilon_pairs",
    "performance_source",
    "robustness_source",
)

OUTPUT_FILES = (
    "portfolio_summary.csv",
    "portfolio_summary.json",
    "final_performance_summary.png",
    "final_robustness_summary.png",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _finite_float(value: Any, label: str) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(numeric_value):
        raise ValueError(f"{label} must be a finite number.")
    return numeric_value


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({field: summary[field] for field in SUMMARY_FIELDS})
    return path


def _epsilon_row(rows: list[dict[str, str]], label: str) -> dict[str, str]:
    for row in rows:
        if row["epsilon_label"] == label:
            return row
    raise ValueError(f"Missing epsilon row: {label}")


def _project_relative(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _batch_summary_lookup(
    benchmark_rows: list[dict[str, str]],
) -> dict[tuple[str, int], dict[str, str]]:
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for row in benchmark_rows:
        if row["suite"] != "batch_size_scaling":
            continue
        lookup[(row["backend"], int(row["batch_size"]))] = row
    return lookup


def build_portfolio_summary(
    *,
    ewp3d_dir: Path = DEFAULT_EWP3D_DIR,
    ewp3e_dir: Path = DEFAULT_EWP3E_DIR,
) -> dict[str, Any]:
    robustness_rows = _read_csv_rows(ewp3e_dir / "robustness_summary.csv")
    metadata = _read_json(ewp3e_dir / "run_metadata.json")
    timing = _read_json(ewp3e_dir / "timing_summary.json")
    crossover = _read_json(ewp3d_dir / "crossover_analysis.json")

    if len(robustness_rows) != 7:
        raise ValueError("Expected 7 robustness epsilon rows.")
    sample_counts = {int(row["total_samples"]) for row in robustness_rows}
    if sample_counts != {10000}:
        raise ValueError(f"Expected only sample_count=10000, got {sample_counts}.")
    if metadata.get("backend") != "cupy":
        raise ValueError("EWP3-E metadata must record backend=cupy.")
    if metadata.get("gpu") != "NVIDIA GeForce RTX 2080 Ti":
        raise ValueError("EWP3-E metadata must record the RTX 2080 Ti GPU.")
    if metadata.get("dataset_checksum_matches") is not True:
        raise ValueError("EWP3-E metadata must record a valid dataset checksum.")

    clean_row = _epsilon_row(robustness_rows, "0")
    row_4 = _epsilon_row(robustness_rows, "4/255")
    row_8 = _epsilon_row(robustness_rows, "8/255")
    row_16 = _epsilon_row(robustness_rows, "16/255")

    summary = {
        "schema_version": 1,
        "full_test_sample_count": int(clean_row["total_samples"]),
        "backend": metadata["backend"],
        "gpu": metadata["gpu"],
        "cupy_version": metadata["cupy_version"],
        "numpy_version": metadata["numpy_version"],
        "python_version": metadata["python_version"],
        "clean_accuracy": _finite_float(clean_row["clean_accuracy"], "clean_accuracy"),
        "adversarial_accuracy_4_255": _finite_float(
            row_4["adversarial_accuracy"], "adversarial_accuracy_4_255"
        ),
        "adversarial_accuracy_8_255": _finite_float(
            row_8["adversarial_accuracy"], "adversarial_accuracy_8_255"
        ),
        "adversarial_accuracy_16_255": _finite_float(
            row_16["adversarial_accuracy"], "adversarial_accuracy_16_255"
        ),
        "attack_success_rate_4_255": _finite_float(
            row_4["attack_success_rate"], "attack_success_rate_4_255"
        ),
        "attack_success_rate_8_255": _finite_float(
            row_8["attack_success_rate"], "attack_success_rate_8_255"
        ),
        "attack_success_rate_16_255": _finite_float(
            row_16["attack_success_rate"], "attack_success_rate_16_255"
        ),
        "first_tested_gpu_faster_batch_size": int(
            crossover["first_gpu_faster_batch_size"]
        ),
        "best_tested_speedup": _finite_float(crossover["max_speedup"], "max_speedup"),
        "best_tested_batch_size": int(crossover["max_speedup_batch_size"]),
        "full_10k_evaluation_throughput": _finite_float(
            timing["evaluation_sample_epsilon_pairs_per_second"],
            "evaluation_sample_epsilon_pairs_per_second",
        ),
        "full_10k_evaluation_wall_seconds": _finite_float(
            timing["evaluation_wall_seconds"],
            "evaluation_wall_seconds",
        ),
        "sample_epsilon_pairs": int(timing["sample_epsilon_pairs"]),
        "speedup_definition": (
            "CPU evaluation_wall_seconds / GPU evaluation_wall_seconds for "
            "matched workloads"
        ),
        "performance_source": _project_relative(ewp3d_dir),
        "robustness_source": _project_relative(ewp3e_dir),
    }
    return summary


def plot_final_performance_summary(
    *,
    ewp3d_dir: Path = DEFAULT_EWP3D_DIR,
    output_path: Path,
) -> Path:
    benchmark_rows = _read_csv_rows(ewp3d_dir / "benchmark_summary.csv")
    speedup_rows = _read_csv_rows(ewp3d_dir / "speedup_summary.csv")
    crossover = _read_json(ewp3d_dir / "crossover_analysis.json")
    lookup = _batch_summary_lookup(benchmark_rows)
    batch_sizes = sorted({int(row["batch_size"]) for row in speedup_rows})
    numpy_throughput = [
        _finite_float(lookup[("numpy", batch)]["throughput_mean"], "numpy throughput")
        for batch in batch_sizes
    ]
    cupy_throughput = [
        _finite_float(lookup[("cupy", batch)]["throughput_mean"], "cupy throughput")
        for batch in batch_sizes
    ]
    speedups = [
        _finite_float(row["evaluation_speedup_median"], "evaluation speedup")
        for row in sorted(speedup_rows, key=lambda row: int(row["batch_size"]))
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(
        batch_sizes,
        numpy_throughput,
        marker="o",
        linewidth=2.2,
        color="#4c78a8",
        label="NumPy",
    )
    axes[0].plot(
        batch_sizes,
        cupy_throughput,
        marker="o",
        linewidth=2.2,
        color="#f58518",
        label="CuPy",
    )
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Sample-epsilon pairs / second")
    axes[0].set_title("Throughput vs Batch Size")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="Backend")

    axes[1].plot(
        batch_sizes,
        speedups,
        marker="o",
        linewidth=2.2,
        color="#54a24b",
        label="Median speedup",
    )
    axes[1].axhline(
        1.0,
        color="0.45",
        linestyle="--",
        linewidth=1.2,
        label="Break-even",
    )
    first_gpu_batch = int(crossover["first_gpu_faster_batch_size"])
    max_batch = int(crossover["max_speedup_batch_size"])
    max_speedup = _finite_float(crossover["max_speedup"], "max speedup")
    axes[1].scatter(
        [first_gpu_batch, max_batch],
        [
            _finite_float(
                crossover["first_gpu_faster_speedup"],
                "first GPU-faster speedup",
            ),
            max_speedup,
        ],
        s=70,
        color="#e45756",
        zorder=4,
    )
    axes[1].annotate(
        f"First GPU-faster: batch {first_gpu_batch}",
        xy=(first_gpu_batch, float(crossover["first_gpu_faster_speedup"])),
        xytext=(first_gpu_batch + 6, 1.25),
        arrowprops={"arrowstyle": "->", "color": "0.35"},
        fontsize=9,
    )
    axes[1].annotate(
        f"Best tested: {max_speedup:.2f}x at batch {max_batch}",
        xy=(max_batch, max_speedup),
        xytext=(max_batch - 72, max_speedup - 0.65),
        arrowprops={"arrowstyle": "->", "color": "0.35"},
        fontsize=9,
    )
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("CPU / GPU evaluation-wall speedup")
    axes[1].set_title("Matched CPU/GPU Speedup")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle(
        "FGSM Evaluation Scaling on RTX 2080 Ti | "
        "1000 samples, epsilons {0, 4/255}, 3 repeats",
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_final_robustness_summary(
    *,
    ewp3e_dir: Path = DEFAULT_EWP3E_DIR,
    output_path: Path,
) -> Path:
    rows = _read_csv_rows(ewp3e_dir / "robustness_summary.csv")
    labels = [row["epsilon_label"] for row in rows]
    x_values = list(range(len(rows)))
    clean_accuracy = [
        100.0 * _finite_float(row["clean_accuracy"], "clean accuracy") for row in rows
    ]
    adversarial_accuracy = [
        100.0 * _finite_float(row["adversarial_accuracy"], "adversarial accuracy")
        for row in rows
    ]
    attack_success_rate = [
        100.0 * _finite_float(row["attack_success_rate"], "attack success rate")
        for row in rows
    ]

    figure, axes = plt.subplots(figsize=(9.8, 5.0))
    axes.plot(
        x_values,
        clean_accuracy,
        marker="o",
        linewidth=2.2,
        color="#4c78a8",
        label="Clean accuracy baseline",
    )
    axes.plot(
        x_values,
        adversarial_accuracy,
        marker="o",
        linewidth=2.2,
        color="#f58518",
        label="Adversarial accuracy",
    )
    axes.plot(
        x_values,
        attack_success_rate,
        marker="o",
        linewidth=2.2,
        color="#e45756",
        label="Attack success rate",
    )
    axes.set_xticks(x_values)
    axes.set_xticklabels(labels)
    axes.set_ylim(-3.0, 103.0)
    axes.set_xlabel("FGSM epsilon")
    axes.set_ylabel("Percent")
    axes.set_title("Full CIFAR-10 Test-Set FGSM Robustness | 10,000 samples")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def build_portfolio_evidence(
    *,
    ewp3d_dir: Path = DEFAULT_EWP3D_DIR,
    ewp3e_dir: Path = DEFAULT_EWP3E_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
) -> dict[str, Path]:
    existing_outputs = [output_dir / filename for filename in OUTPUT_FILES]
    existing_outputs = [path for path in existing_outputs if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Portfolio evidence already exists. Pass --overwrite to regenerate: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_portfolio_summary(ewp3d_dir=ewp3d_dir, ewp3e_dir=ewp3e_dir)
    summary_csv = _write_summary_csv(output_dir / "portfolio_summary.csv", summary)
    summary_json = save_metrics(summary, output_dir / "portfolio_summary.json")
    performance_plot = plot_final_performance_summary(
        ewp3d_dir=ewp3d_dir,
        output_path=output_dir / "final_performance_summary.png",
    )
    robustness_plot = plot_final_robustness_summary(
        ewp3e_dir=ewp3e_dir,
        output_path=output_dir / "final_robustness_summary.png",
    )
    return {
        "output_dir": output_dir,
        "portfolio_summary_csv": summary_csv,
        "portfolio_summary_json": summary_json,
        "final_performance_summary": performance_plot,
        "final_robustness_summary": robustness_plot,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate final portfolio figures and summary tables from curated "
            "EWP3-D/EWP3-E artifacts."
        )
    )
    parser.add_argument("--ewp3d-dir", type=Path, default=DEFAULT_EWP3D_DIR)
    parser.add_argument("--ewp3e-dir", type=Path, default=DEFAULT_EWP3E_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate portfolio evidence if outputs already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_portfolio_evidence(
        ewp3d_dir=args.ewp3d_dir,
        ewp3e_dir=args.ewp3e_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
