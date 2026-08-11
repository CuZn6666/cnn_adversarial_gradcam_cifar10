"""Curate plots and summaries from a saved FGSM runner artifact directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.default_config import PROJECT_ROOT
from experiments.fgsm.run_fgsm_experiment import parse_epsilon_values
from src.metrics import save_metrics
from src.plotting import (
    plot_fgsm_accuracy_drop_vs_epsilon,
    plot_fgsm_attack_success_rate_vs_epsilon,
    plot_fgsm_portfolio_accuracy_vs_epsilon,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "curated" / "ewp3c"
DEFAULT_INTERPRETATION = (
    "Medium-scale sanity run; use for runner stability and artifact quality, "
    "not as the final full CIFAR-10 robustness conclusion."
)
REQUIRED_JSON_ARTIFACTS = (
    "config.json",
    "environment.json",
    "metrics.json",
    "timing.json",
    "summary.json",
    "status.json",
)
REQUIRED_ARTIFACTS = (*REQUIRED_JSON_ARTIFACTS, "metrics.csv")
ROBUSTNESS_SUMMARY_FIELDS = (
    "epsilon",
    "epsilon_label",
    "total_samples",
    "clean_correct",
    "adversarial_correct",
    "clean_correct_samples",
    "successful_attacks",
    "clean_accuracy",
    "adversarial_accuracy",
    "accuracy_drop",
    "attack_success_rate",
)
CURATED_OUTPUT_FILES = (
    "robustness_summary.csv",
    "timing_summary.json",
    "run_metadata.json",
    "accuracy_vs_epsilon.png",
    "attack_success_rate_vs_epsilon.png",
    "accuracy_drop_vs_epsilon.png",
    "runtime_throughput_summary.png",
)


def load_json(path: str | Path) -> Any:
    artifact_path = Path(path)
    try:
        with artifact_path.open(encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Required artifact is missing: {artifact_path}")
    except json.JSONDecodeError as error:
        raise ValueError(f"Artifact is not valid JSON: {artifact_path}") from error


def load_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    artifacts: dict[str, Any] = {"run_dir": run_path}
    for filename in REQUIRED_ARTIFACTS:
        artifact_path = run_path / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Required artifact is missing: {artifact_path}")
    for filename in REQUIRED_JSON_ARTIFACTS:
        artifacts[filename.removesuffix(".json")] = load_json(run_path / filename)
    return artifacts


def epsilon_label(epsilon: float) -> str:
    value = float(epsilon)
    if not math.isfinite(value):
        raise ValueError("epsilon values must be finite.")
    if value == 0.0:
        return "0"
    scaled = value * 255.0
    rounded = round(scaled)
    if math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1e-10):
        return f"{rounded}/255"
    return f"{value:g}"


def epsilon_labels(rows: list[dict[str, Any]]) -> list[str]:
    return [epsilon_label(float(row["epsilon"])) for row in rows]


def _require_finite_number(value: Any, label: str) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(numeric_value):
        raise ValueError(f"{label} must be a finite number.")
    return numeric_value


def _require_positive_number(value: Any, label: str) -> float:
    numeric_value = _require_finite_number(value, label)
    if numeric_value <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return numeric_value


def validate_run_artifacts(
    artifacts: dict[str, Any],
    *,
    expected_sample_count: int | None = None,
    expected_epsilons: tuple[float, ...] | None = None,
    expected_backend: str | None = None,
    expected_gpu_name: str | None = None,
) -> list[dict[str, Any]]:
    status = artifacts["status"]
    if status.get("status") != "COMPLETED":
        raise ValueError("Run status must be COMPLETED before curation.")

    metrics = artifacts["metrics"]
    rows = metrics.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("metrics.json must contain a non-empty results list.")

    config = artifacts["config"]
    if expected_backend is not None and config.get("backend") != expected_backend:
        raise ValueError(
            f"Expected backend {expected_backend}, observed {config.get('backend')}."
        )
    configured_epsilons = [float(value) for value in config["epsilon_values"]]
    observed_epsilons = [float(row["epsilon"]) for row in rows]
    if observed_epsilons != configured_epsilons:
        raise ValueError("Metric epsilon order must match config epsilon order.")
    if expected_epsilons is not None and observed_epsilons != list(expected_epsilons):
        raise ValueError("Metric epsilon order does not match expected epsilons.")

    summary = artifacts["summary"]
    dataset = summary.get("dataset", {})
    if dataset.get("archive_checksum_matches") is not True:
        raise ValueError("Dataset checksum must be validated before curation.")

    sample_counts = {int(row["total_samples"]) for row in rows}
    if len(sample_counts) != 1:
        raise ValueError("All epsilon rows must use the same sample count.")
    sample_count = sample_counts.pop()
    if sample_count <= 0:
        raise ValueError("sample count must be positive.")
    if expected_sample_count is not None and sample_count != expected_sample_count:
        raise ValueError(
            f"Expected {expected_sample_count} samples, observed {sample_count}."
        )
    if dataset.get("evaluated_samples") != sample_count:
        raise ValueError("summary.json dataset evaluated_samples must match metrics.")

    if expected_gpu_name is not None:
        environment = artifacts["environment"]
        observed_gpu_name = environment.get("cupy", {}).get("gpu_name")
        if observed_gpu_name != expected_gpu_name:
            raise ValueError(
                f"Expected GPU {expected_gpu_name}, observed {observed_gpu_name}."
            )

    for index, row in enumerate(rows):
        for field in ROBUSTNESS_SUMMARY_FIELDS:
            if field == "epsilon_label":
                continue
            if field not in row:
                raise ValueError(f"metrics row {index} is missing {field}.")
        for field in (
            "epsilon",
            "clean_accuracy",
            "adversarial_accuracy",
            "accuracy_drop",
            "attack_success_rate",
        ):
            _require_finite_number(row[field], f"metrics row {index} {field}")
        for field in (
            "clean_accuracy",
            "adversarial_accuracy",
            "attack_success_rate",
        ):
            value = float(row[field])
            if value < 0.0 or value > 1.0:
                raise ValueError(f"metrics row {index} {field} must be in [0, 1].")
        for field in (
            "total_samples",
            "clean_correct",
            "adversarial_correct",
            "clean_correct_samples",
            "successful_attacks",
        ):
            value = row[field]
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"metrics row {index} {field} must be non-negative.")

    epsilon_zero_rows = [
        row for row in rows if math.isclose(float(row["epsilon"]), 0.0, abs_tol=0.0)
    ]
    if epsilon_zero_rows:
        epsilon_zero = epsilon_zero_rows[0]
        if int(epsilon_zero["clean_correct"]) != int(
            epsilon_zero["adversarial_correct"]
        ):
            raise ValueError(
                "epsilon=0 clean_correct must match adversarial_correct."
            )
        if int(epsilon_zero["successful_attacks"]) != 0:
            raise ValueError("epsilon=0 successful_attacks must be zero.")
        if not math.isclose(
            float(epsilon_zero["clean_accuracy"]),
            float(epsilon_zero["adversarial_accuracy"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "epsilon=0 clean_accuracy must match adversarial_accuracy."
            )

    timing = artifacts["timing"]
    _require_positive_number(timing.get("total_wall_seconds"), "total_wall_seconds")
    _require_positive_number(
        timing.get("evaluation_wall_seconds"),
        "evaluation_wall_seconds",
    )
    _require_positive_number(
        timing.get("sample_epsilon_pairs"),
        "sample_epsilon_pairs",
    )
    _require_positive_number(
        timing.get("evaluation_sample_epsilon_pairs_per_second"),
        "evaluation_sample_epsilon_pairs_per_second",
    )

    return rows


def _write_robustness_summary(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = epsilon_labels(rows)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ROBUSTNESS_SUMMARY_FIELDS)
        writer.writeheader()
        for row, label in zip(rows, labels):
            writer.writerow(
                {
                    field: label if field == "epsilon_label" else row[field]
                    for field in ROBUSTNESS_SUMMARY_FIELDS
                }
            )
    return output_path


def build_run_metadata(
    artifacts: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    interpretation: str = DEFAULT_INTERPRETATION,
) -> dict[str, Any]:
    config = artifacts["config"]
    environment = artifacts["environment"]
    summary = artifacts["summary"]
    timing = artifacts["timing"]
    cupy = environment.get("cupy", {})
    git = environment.get("git", {})
    dataset = summary.get("dataset", {})

    return {
        "schema_version": 1,
        "run_id": config["run_id"],
        "backend": config["backend"],
        "gpu": cupy.get("gpu_name"),
        "cupy_version": cupy.get("version"),
        "cuda_runtime_version": cupy.get("cuda_runtime_version"),
        "python_version": environment.get("python_version"),
        "numpy_version": environment.get("numpy_version"),
        "hostname": environment.get("hostname"),
        "checkpoint_path": config["checkpoint_path"],
        "dataset_split": config["split"],
        "dataset_checksum_matches": dataset.get("archive_checksum_matches"),
        "sample_count": rows[0]["total_samples"],
        "batch_size": config["batch_size"],
        "epsilon_values": [row["epsilon"] for row in rows],
        "epsilon_labels": epsilon_labels(rows),
        "timing_method": timing.get("timing_method"),
        "gpu_synchronization": timing.get("gpu_synchronization"),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "seed": config["seed"],
        "interpretation": interpretation,
    }


def build_timing_summary(
    artifacts: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    timing = artifacts["timing"]
    config = artifacts["config"]
    return {
        "schema_version": 1,
        "run_id": config["run_id"],
        "backend": config["backend"],
        "sample_count": rows[0]["total_samples"],
        "epsilon_count": len(rows),
        "sample_epsilon_pairs": timing["sample_epsilon_pairs"],
        "total_wall_seconds": timing["total_wall_seconds"],
        "evaluation_wall_seconds": timing["evaluation_wall_seconds"],
        "evaluation_sample_epsilon_pairs_per_second": timing[
            "evaluation_sample_epsilon_pairs_per_second"
        ],
        "timing_method": timing["timing_method"],
        "gpu_synchronization": timing["gpu_synchronization"],
    }


def plot_runtime_throughput_summary(
    timing_summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    runtime_labels = ["Total", "Evaluation"]
    runtime_values = [
        timing_summary["total_wall_seconds"],
        timing_summary["evaluation_wall_seconds"],
    ]
    throughput = timing_summary["evaluation_sample_epsilon_pairs_per_second"]

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(runtime_labels, runtime_values, color=["#4c78a8", "#f58518"])
    axes[0].set_ylabel("Wall time (seconds)")
    axes[0].set_title("Runtime")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        ["Evaluation"],
        [throughput],
        color="#54a24b",
    )
    axes[1].set_ylabel("Sample-epsilon pairs / second")
    axes[1].set_title("Throughput")
    axes[1].grid(axis="y", alpha=0.3)

    figure.suptitle(
        "FGSM Runner Timing | "
        f"{timing_summary['backend']} | "
        f"{timing_summary['sample_count']} samples | "
        f"{timing_summary['epsilon_count']} epsilons",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def _prepare_output_dir(
    output_root: Path,
    run_id: str,
    *,
    overwrite: bool,
) -> Path:
    output_dir = output_root / run_id
    existing_outputs = [
        output_dir / filename
        for filename in CURATED_OUTPUT_FILES
        if (output_dir / filename).exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Curated output already exists. Pass --overwrite to regenerate: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def curate_fgsm_run(
    run_dir: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    expected_sample_count: int | None = None,
    expected_epsilons: tuple[float, ...] | None = None,
    expected_backend: str | None = None,
    expected_gpu_name: str | None = None,
    interpretation: str = DEFAULT_INTERPRETATION,
    overwrite: bool = False,
) -> dict[str, Path]:
    artifacts = load_run_artifacts(run_dir)
    rows = validate_run_artifacts(
        artifacts,
        expected_sample_count=expected_sample_count,
        expected_epsilons=expected_epsilons,
        expected_backend=expected_backend,
        expected_gpu_name=expected_gpu_name,
    )
    run_id = artifacts["config"]["run_id"]
    output_dir = _prepare_output_dir(
        Path(output_root),
        run_id,
        overwrite=overwrite,
    )

    labels = epsilon_labels(rows)
    robustness_summary_path = _write_robustness_summary(
        rows,
        output_dir / "robustness_summary.csv",
    )
    run_metadata_path = save_metrics(
        build_run_metadata(artifacts, rows, interpretation=interpretation),
        output_dir / "run_metadata.json",
    )
    timing_summary = build_timing_summary(artifacts, rows)
    timing_summary_path = save_metrics(
        timing_summary,
        output_dir / "timing_summary.json",
    )
    accuracy_plot_path = plot_fgsm_portfolio_accuracy_vs_epsilon(
        rows,
        output_dir / "accuracy_vs_epsilon.png",
        epsilon_labels=labels,
    )
    attack_success_plot_path = plot_fgsm_attack_success_rate_vs_epsilon(
        rows,
        output_dir / "attack_success_rate_vs_epsilon.png",
        epsilon_labels=labels,
    )
    accuracy_drop_plot_path = plot_fgsm_accuracy_drop_vs_epsilon(
        rows,
        output_dir / "accuracy_drop_vs_epsilon.png",
        epsilon_labels=labels,
    )
    runtime_plot_path = plot_runtime_throughput_summary(
        timing_summary,
        output_dir / "runtime_throughput_summary.png",
    )

    return {
        "output_dir": output_dir,
        "robustness_summary": robustness_summary_path,
        "run_metadata": run_metadata_path,
        "timing_summary": timing_summary_path,
        "accuracy_vs_epsilon": accuracy_plot_path,
        "attack_success_rate_vs_epsilon": attack_success_plot_path,
        "accuracy_drop_vs_epsilon": accuracy_drop_plot_path,
        "runtime_throughput_summary": runtime_plot_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create curated summaries and plots from one FGSM runner artifact "
            "directory."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Raw FGSM run directory containing config/metrics/timing artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Curated output root. A run_id subdirectory is created below it.",
    )
    parser.add_argument(
        "--expected-sample-count",
        type=int,
        default=None,
        help="Optional required sample count for validation.",
    )
    parser.add_argument(
        "--expected-epsilons",
        default=None,
        help="Optional comma-separated expected epsilon list.",
    )
    parser.add_argument(
        "--expected-backend",
        choices=("numpy", "cupy"),
        default=None,
        help="Optional required backend recorded by the run config.",
    )
    parser.add_argument(
        "--expected-gpu-name",
        default=None,
        help="Optional required GPU name recorded by environment metadata.",
    )
    parser.add_argument(
        "--interpretation",
        default=DEFAULT_INTERPRETATION,
        help="Interpretation text to store in run_metadata.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate curated files if they already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expected_epsilons = (
        None
        if args.expected_epsilons is None
        else parse_epsilon_values(args.expected_epsilons)
    )
    outputs = curate_fgsm_run(
        args.run_dir,
        args.output_root,
        expected_sample_count=args.expected_sample_count,
        expected_epsilons=expected_epsilons,
        expected_backend=args.expected_backend,
        expected_gpu_name=args.expected_gpu_name,
        interpretation=args.interpretation,
        overwrite=args.overwrite,
    )
    print(f"curated_output_dir: {outputs['output_dir']}")
    for name, path in outputs.items():
        if name == "output_dir":
            continue
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
