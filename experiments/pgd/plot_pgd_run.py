"""Curate summaries and plots from a saved PGD runner artifact directory."""

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
from experiments.pgd.run_pgd_experiment import parse_single_epsilon
from src.metrics import save_metrics


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "curated" / "ewp4c"
DEFAULT_INTERPRETATION = (
    "Small PGD cluster smoke run; use for runner and artifact validation, "
    "not as final PGD robustness evidence."
)
DEFAULT_PLOT_FILENAME = "pgd_smoke_summary.png"
DEFAULT_PLOT_TITLE = "PGD-Linf Runner Smoke"
REQUIRED_JSON_ARTIFACTS = (
    "config.json",
    "environment.json",
    "metrics.json",
    "timing.json",
    "summary.json",
    "status.json",
)
REQUIRED_ARTIFACTS = (*REQUIRED_JSON_ARTIFACTS, "metrics.csv")
PGD_SUMMARY_FIELDS = (
    "epsilon",
    "alpha",
    "steps",
    "random_start",
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
    DEFAULT_PLOT_FILENAME,
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


def _require_non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer.")
    try:
        integer_value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a non-negative integer.") from error
    if integer_value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return integer_value


def validate_run_artifacts(
    artifacts: dict[str, Any],
    *,
    expected_sample_count: int | None = None,
    expected_epsilon: float | None = None,
    expected_alpha: float | None = None,
    expected_steps: int | None = None,
    expected_random_start: bool | None = None,
    expected_seed: int | None = None,
    expected_split: str | None = None,
    expected_checkpoint_path: str | None = None,
    expected_backend: str | None = None,
    expected_gpu_name: str | None = None,
) -> dict[str, Any]:
    status = artifacts["status"]
    if status.get("status") != "COMPLETED":
        raise ValueError("Run status must be COMPLETED before curation.")

    config = artifacts["config"]
    if config.get("attack") != "pgd_linf":
        raise ValueError("config.json must describe a pgd_linf run.")
    if expected_backend is not None and config.get("backend") != expected_backend:
        raise ValueError(
            f"Expected backend {expected_backend}, observed {config.get('backend')}."
        )
    if expected_epsilon is not None and not math.isclose(
        float(config.get("epsilon")),
        expected_epsilon,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Configured epsilon does not match expected epsilon.")
    if expected_alpha is not None and not math.isclose(
        float(config.get("alpha")),
        expected_alpha,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Configured alpha does not match expected alpha.")
    if expected_steps is not None and int(config.get("steps")) != expected_steps:
        raise ValueError("Configured steps does not match expected steps.")
    if (
        expected_random_start is not None
        and config.get("random_start") is not expected_random_start
    ):
        raise ValueError(
            "Configured random_start does not match expected random_start."
        )
    if expected_seed is not None and int(config.get("seed")) != expected_seed:
        raise ValueError("Configured seed does not match expected seed.")
    if expected_split is not None and config.get("split") != expected_split:
        raise ValueError("Configured split does not match expected split.")
    if (
        expected_checkpoint_path is not None
        and config.get("checkpoint_path") != expected_checkpoint_path
    ):
        raise ValueError(
            "Configured checkpoint_path does not match expected checkpoint_path."
        )

    metrics = artifacts["metrics"]
    rows = metrics.get("results")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("PGD EWP4-C metrics.json must contain one results row.")
    row = rows[0]
    if row.get("attack") != "pgd_linf":
        raise ValueError("metrics row must describe attack=pgd_linf.")

    summary = artifacts["summary"]
    if summary.get("attack") != "pgd_linf":
        raise ValueError("summary.json must describe attack=pgd_linf.")
    dataset = summary.get("dataset", {})
    if dataset.get("archive_checksum_matches") is not True:
        raise ValueError("Dataset checksum must be validated before curation.")

    if expected_gpu_name is not None:
        environment = artifacts["environment"]
        observed_gpu_name = environment.get("cupy", {}).get("gpu_name")
        if observed_gpu_name != expected_gpu_name:
            raise ValueError(
                f"Expected GPU {expected_gpu_name}, observed {observed_gpu_name}."
            )

    for field in PGD_SUMMARY_FIELDS:
        if field not in row:
            raise ValueError(f"metrics row is missing {field}.")
    for field in (
        "epsilon",
        "alpha",
        "clean_accuracy",
        "adversarial_accuracy",
        "accuracy_drop",
        "attack_success_rate",
    ):
        _require_finite_number(row[field], f"metrics row {field}")
    for field in (
        "clean_accuracy",
        "adversarial_accuracy",
        "attack_success_rate",
    ):
        value = float(row[field])
        if value < 0.0 or value > 1.0:
            raise ValueError(f"metrics row {field} must be in [0, 1].")
    for field in (
        "total_samples",
        "clean_correct",
        "adversarial_correct",
        "clean_correct_samples",
        "successful_attacks",
        "steps",
    ):
        _require_non_negative_int(row[field], f"metrics row {field}")
    if not isinstance(row["random_start"], bool):
        raise ValueError("metrics row random_start must be a boolean.")

    sample_count = int(row["total_samples"])
    if sample_count <= 0:
        raise ValueError("sample count must be positive.")
    if expected_sample_count is not None and sample_count != expected_sample_count:
        raise ValueError(
            f"Expected {expected_sample_count} samples, observed {sample_count}."
        )
    if int(config["max_samples"]) != sample_count:
        raise ValueError("config max_samples must match metrics total_samples.")
    if int(summary.get("sample_count")) != sample_count:
        raise ValueError("summary sample_count must match metrics total_samples.")
    if dataset.get("evaluated_samples") != sample_count:
        raise ValueError("summary dataset evaluated_samples must match metrics.")

    for field in ("epsilon", "alpha"):
        if not math.isclose(
            float(config[field]),
            float(row[field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"config {field} must match metrics row {field}.")
    for field in ("steps", "random_start"):
        if config[field] != row[field]:
            raise ValueError(f"config {field} must match metrics row {field}.")

    clean_correct = int(row["clean_correct"])
    adversarial_correct = int(row["adversarial_correct"])
    clean_correct_samples = int(row["clean_correct_samples"])
    successful_attacks = int(row["successful_attacks"])
    for field_name, value in (
        ("clean_correct", clean_correct),
        ("adversarial_correct", adversarial_correct),
        ("clean_correct_samples", clean_correct_samples),
    ):
        if value > sample_count:
            raise ValueError(f"{field_name} must not exceed total_samples.")
    if clean_correct_samples != clean_correct:
        raise ValueError("clean_correct_samples must equal clean_correct.")
    if successful_attacks > clean_correct_samples:
        raise ValueError(
            "successful_attacks must not exceed clean_correct_samples."
        )
    if not math.isclose(
        float(row["clean_accuracy"]),
        clean_correct / sample_count,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("clean_accuracy must match clean_correct / total_samples.")
    if not math.isclose(
        float(row["adversarial_accuracy"]),
        adversarial_correct / sample_count,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "adversarial_accuracy must match adversarial_correct / total_samples."
        )
    if not math.isclose(
        float(row["accuracy_drop"]),
        float(row["clean_accuracy"]) - float(row["adversarial_accuracy"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "accuracy_drop must equal clean_accuracy - adversarial_accuracy."
        )
    expected_asr = (
        successful_attacks / clean_correct_samples
        if clean_correct_samples > 0
        else 0.0
    )
    if not math.isclose(
        float(row["attack_success_rate"]),
        expected_asr,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "attack_success_rate must match successful_attacks / "
            "clean_correct_samples."
        )

    timing = artifacts["timing"]
    _require_positive_number(timing.get("total_wall_seconds"), "total_wall_seconds")
    _require_positive_number(
        timing.get("evaluation_wall_seconds"),
        "evaluation_wall_seconds",
    )
    _require_positive_number(timing.get("sample_count"), "timing sample_count")
    if int(timing["sample_count"]) != sample_count:
        raise ValueError("timing sample_count must match metrics total_samples.")
    if int(timing.get("pgd_steps")) != int(row["steps"]):
        raise ValueError("timing pgd_steps must match metrics row steps.")
    if int(timing.get("sample_steps")) != sample_count * int(row["steps"]):
        raise ValueError("timing sample_steps must equal sample_count * steps.")
    if int(timing.get("gradient_evaluations")) != sample_count * int(row["steps"]):
        raise ValueError(
            "timing gradient_evaluations must equal sample_count * steps."
        )
    _require_positive_number(
        timing.get("samples_per_second"),
        "samples_per_second",
    )
    if int(row["steps"]) > 0:
        _require_positive_number(
            timing.get("sample_steps_per_second"),
            "sample_steps_per_second",
        )

    return row


def _write_pgd_summary(row: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PGD_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({field: row[field] for field in PGD_SUMMARY_FIELDS})
    return output_path


def build_run_metadata(
    artifacts: dict[str, Any],
    row: dict[str, Any],
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
        "experiment_type": "pgd_linf",
        "attack": "pgd_linf",
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
        "sample_count": row["total_samples"],
        "batch_size": config["batch_size"],
        "epsilon": row["epsilon"],
        "alpha": row["alpha"],
        "steps": row["steps"],
        "random_start": row["random_start"],
        "timing_method": timing.get("timing_method"),
        "gpu_synchronization": timing.get("gpu_synchronization"),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "seed": config["seed"],
        "random_start_seed_strategy": config.get("random_start_seed_strategy"),
        "interpretation": interpretation,
    }


def build_timing_summary(
    artifacts: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    timing = artifacts["timing"]
    config = artifacts["config"]
    return {
        "schema_version": 1,
        "run_id": config["run_id"],
        "attack": "pgd_linf",
        "backend": config["backend"],
        "sample_count": row["total_samples"],
        "pgd_steps": row["steps"],
        "sample_steps": timing["sample_steps"],
        "gradient_evaluations": timing["gradient_evaluations"],
        "total_wall_seconds": timing["total_wall_seconds"],
        "evaluation_wall_seconds": timing["evaluation_wall_seconds"],
        "samples_per_second": timing["samples_per_second"],
        "sample_steps_per_second": timing["sample_steps_per_second"],
        "timing_method": timing["timing_method"],
        "gpu_synchronization": timing["gpu_synchronization"],
    }


def plot_pgd_smoke_summary(
    row: dict[str, Any],
    timing_summary: dict[str, Any],
    output_path: str | Path,
    *,
    backend: str,
    title: str = DEFAULT_PLOT_TITLE,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metric_labels = [
        "Clean accuracy",
        "PGD accuracy",
        "Attack success rate",
    ]
    metric_values = [
        float(row["clean_accuracy"]),
        float(row["adversarial_accuracy"]),
        float(row["attack_success_rate"]),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    bars = axes[0].bar(
        metric_labels,
        metric_values,
        color=["#4c78a8", "#f58518", "#e45756"],
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("PGD robustness metrics")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, metric_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axes[1].bar(
        ["Samples/s", "Sample-steps/s"],
        [
            float(timing_summary["samples_per_second"]),
            float(timing_summary["sample_steps_per_second"]),
        ],
        color=["#54a24b", "#72b7b2"],
    )
    axes[1].set_ylabel("Throughput")
    axes[1].set_title("Evaluation throughput")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=20)

    figure.suptitle(
        f"{title} | "
        f"{backend} | "
        f"{row['total_samples']} samples | "
        f"eps={float(row['epsilon']) * 255:g}/255 | "
        f"alpha={float(row['alpha']) * 255:g}/255 | "
        f"{row['steps']} steps",
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
    plot_filename: str = DEFAULT_PLOT_FILENAME,
) -> Path:
    output_dir = output_root / run_id
    expected_filenames = {
        "robustness_summary.csv",
        "timing_summary.json",
        "run_metadata.json",
        plot_filename,
    }
    existing_outputs = [
        output_dir / filename
        for filename in expected_filenames
        if (output_dir / filename).exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Curated output already exists. Pass --overwrite to regenerate: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def curate_pgd_run(
    run_dir: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    expected_sample_count: int | None = None,
    expected_epsilon: float | None = None,
    expected_alpha: float | None = None,
    expected_steps: int | None = None,
    expected_random_start: bool | None = None,
    expected_seed: int | None = None,
    expected_split: str | None = None,
    expected_checkpoint_path: str | None = None,
    expected_backend: str | None = None,
    expected_gpu_name: str | None = None,
    interpretation: str = DEFAULT_INTERPRETATION,
    plot_filename: str = DEFAULT_PLOT_FILENAME,
    plot_title: str = DEFAULT_PLOT_TITLE,
    overwrite: bool = False,
) -> dict[str, Path]:
    artifacts = load_run_artifacts(run_dir)
    row = validate_run_artifacts(
        artifacts,
        expected_sample_count=expected_sample_count,
        expected_epsilon=expected_epsilon,
        expected_alpha=expected_alpha,
        expected_steps=expected_steps,
        expected_random_start=expected_random_start,
        expected_seed=expected_seed,
        expected_split=expected_split,
        expected_checkpoint_path=expected_checkpoint_path,
        expected_backend=expected_backend,
        expected_gpu_name=expected_gpu_name,
    )
    run_id = artifacts["config"]["run_id"]
    output_dir = _prepare_output_dir(
        Path(output_root),
        run_id,
        overwrite=overwrite,
        plot_filename=plot_filename,
    )

    robustness_summary_path = _write_pgd_summary(
        row,
        output_dir / "robustness_summary.csv",
    )
    run_metadata_path = save_metrics(
        build_run_metadata(artifacts, row, interpretation=interpretation),
        output_dir / "run_metadata.json",
    )
    timing_summary = build_timing_summary(artifacts, row)
    timing_summary_path = save_metrics(
        timing_summary,
        output_dir / "timing_summary.json",
    )
    plot_path = plot_pgd_smoke_summary(
        row,
        timing_summary,
        output_dir / plot_filename,
        backend=artifacts["config"]["backend"],
        title=plot_title,
    )

    outputs = {
        "output_dir": output_dir,
        "robustness_summary": robustness_summary_path,
        "run_metadata": run_metadata_path,
        "timing_summary": timing_summary_path,
        "pgd_summary_plot": plot_path,
    }
    outputs[Path(plot_filename).stem] = plot_path
    if plot_filename == DEFAULT_PLOT_FILENAME:
        outputs["pgd_smoke_summary"] = plot_path
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create curated summaries and a compact plot from one PGD runner "
            "artifact directory."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Raw PGD run directory containing config/metrics/timing artifacts.",
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
        "--expected-epsilon",
        default=None,
        help="Optional required PGD epsilon value as a float or fraction.",
    )
    parser.add_argument(
        "--expected-alpha",
        default=None,
        help="Optional required PGD alpha value as a float or fraction.",
    )
    parser.add_argument(
        "--expected-steps",
        type=int,
        default=None,
        help="Optional required PGD step count.",
    )
    parser.add_argument(
        "--expected-random-start",
        choices=("true", "false"),
        default=None,
        help="Optional required random-start setting.",
    )
    parser.add_argument(
        "--expected-seed",
        type=int,
        default=None,
        help="Optional required run seed.",
    )
    parser.add_argument(
        "--expected-split",
        choices=("train", "test"),
        default=None,
        help="Optional required CIFAR-10 split.",
    )
    parser.add_argument(
        "--expected-checkpoint",
        default=None,
        help="Optional required checkpoint path recorded in config.json.",
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
        "--plot-filename",
        default=DEFAULT_PLOT_FILENAME,
        help="Output PNG filename for the curated PGD summary plot.",
    )
    parser.add_argument(
        "--plot-title",
        default=DEFAULT_PLOT_TITLE,
        help="Title prefix for the curated PGD summary plot.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate curated files if they already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expected_epsilon = (
        None
        if args.expected_epsilon is None
        else parse_single_epsilon(args.expected_epsilon, name="expected_epsilon")
    )
    expected_alpha = (
        None
        if args.expected_alpha is None
        else parse_single_epsilon(args.expected_alpha, name="expected_alpha")
    )
    expected_random_start = (
        None
        if args.expected_random_start is None
        else args.expected_random_start == "true"
    )
    outputs = curate_pgd_run(
        args.run_dir,
        args.output_root,
        expected_sample_count=args.expected_sample_count,
        expected_epsilon=expected_epsilon,
        expected_alpha=expected_alpha,
        expected_steps=args.expected_steps,
        expected_random_start=expected_random_start,
        expected_seed=args.expected_seed,
        expected_split=args.expected_split,
        expected_checkpoint_path=args.expected_checkpoint,
        expected_backend=args.expected_backend,
        expected_gpu_name=args.expected_gpu_name,
        interpretation=args.interpretation,
        plot_filename=args.plot_filename,
        plot_title=args.plot_title,
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
