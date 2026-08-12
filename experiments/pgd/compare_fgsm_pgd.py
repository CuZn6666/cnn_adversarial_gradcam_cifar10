"""Build curated FGSM-vs-PGD comparison artifacts."""

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
from experiments.fgsm.plot_fgsm_run import epsilon_label
from experiments.pgd.run_pgd_experiment import parse_single_epsilon
from src.metrics import save_metrics


DEFAULT_FGSM_CURATED_DIR = (
    PROJECT_ROOT
    / "results"
    / "curated"
    / "ewp3e"
    / "20260812T115232600695Z_fgsm_cupy"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "curated" / "portfolio"
DEFAULT_EPSILON = 8.0 / 255.0
OUTPUT_FILES = (
    "fgsm_vs_pgd_summary.csv",
    "fgsm_vs_pgd_summary.json",
    "final_fgsm_vs_pgd_summary.png",
)
SUMMARY_FIELDS = (
    "attack",
    "run_id",
    "epsilon",
    "epsilon_label",
    "sample_count",
    "clean_accuracy",
    "adversarial_accuracy",
    "accuracy_drop",
    "attack_success_rate",
    "total_samples",
    "clean_correct",
    "adversarial_correct",
    "clean_correct_samples",
    "successful_attacks",
    "source_dir",
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


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    artifact_path = Path(path)
    try:
        with artifact_path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required artifact is missing: {artifact_path}")


def _require_finite_number(value: Any, label: str) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(numeric_value):
        raise ValueError(f"{label} must be a finite number.")
    return numeric_value


def _require_rate(value: Any, label: str) -> float:
    rate = _require_finite_number(value, label)
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"{label} must be in [0, 1].")
    return rate


def _row_for_epsilon(
    rows: list[dict[str, str]],
    epsilon: float,
    *,
    source_label: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if math.isclose(
            float(row["epsilon"]),
            epsilon,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{source_label} must contain exactly one row for epsilon={epsilon}."
        )
    return matches[0]


def _validate_metric_row(
    row: dict[str, Any],
    *,
    expected_sample_count: int,
    source_label: str,
) -> dict[str, Any]:
    total_samples = int(row["total_samples"])
    clean_correct = int(row["clean_correct"])
    adversarial_correct = int(row["adversarial_correct"])
    clean_correct_samples = int(row["clean_correct_samples"])
    successful_attacks = int(row["successful_attacks"])
    if total_samples != expected_sample_count:
        raise ValueError(
            f"{source_label} sample count mismatch: "
            f"expected {expected_sample_count}, observed {total_samples}."
        )
    for field_name, value in (
        ("clean_correct", clean_correct),
        ("adversarial_correct", adversarial_correct),
        ("clean_correct_samples", clean_correct_samples),
    ):
        if value < 0 or value > total_samples:
            raise ValueError(f"{source_label} {field_name} is out of range.")
    if clean_correct_samples != clean_correct:
        raise ValueError(
            f"{source_label} clean_correct_samples must equal clean_correct."
        )
    if successful_attacks < 0 or successful_attacks > clean_correct_samples:
        raise ValueError(f"{source_label} successful_attacks is out of range.")

    clean_accuracy = _require_rate(row["clean_accuracy"], f"{source_label} clean_accuracy")
    adversarial_accuracy = _require_rate(
        row["adversarial_accuracy"],
        f"{source_label} adversarial_accuracy",
    )
    attack_success_rate = _require_rate(
        row["attack_success_rate"],
        f"{source_label} attack_success_rate",
    )
    accuracy_drop = _require_finite_number(
        row["accuracy_drop"],
        f"{source_label} accuracy_drop",
    )

    if not math.isclose(
        clean_accuracy,
        clean_correct / total_samples,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{source_label} clean accuracy is inconsistent.")
    if not math.isclose(
        adversarial_accuracy,
        adversarial_correct / total_samples,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{source_label} adversarial accuracy is inconsistent.")
    if not math.isclose(
        accuracy_drop,
        clean_accuracy - adversarial_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{source_label} accuracy_drop is inconsistent.")
    expected_asr = (
        successful_attacks / clean_correct_samples
        if clean_correct_samples > 0
        else 0.0
    )
    if not math.isclose(
        attack_success_rate,
        expected_asr,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{source_label} attack_success_rate is inconsistent.")

    return {
        "total_samples": total_samples,
        "clean_correct": clean_correct,
        "adversarial_correct": adversarial_correct,
        "clean_correct_samples": clean_correct_samples,
        "successful_attacks": successful_attacks,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "accuracy_drop": accuracy_drop,
        "attack_success_rate": attack_success_rate,
    }


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    expected_backend: str,
    expected_gpu_name: str,
    expected_sample_count: int,
    expected_split: str,
    expected_checkpoint_path: str,
    source_label: str,
) -> None:
    if metadata.get("backend") != expected_backend:
        raise ValueError(
            f"{source_label} backend mismatch: expected {expected_backend}, "
            f"observed {metadata.get('backend')}."
        )
    if metadata.get("gpu") != expected_gpu_name:
        raise ValueError(
            f"{source_label} GPU mismatch: expected {expected_gpu_name}, "
            f"observed {metadata.get('gpu')}."
        )
    if int(metadata.get("sample_count")) != expected_sample_count:
        raise ValueError(f"{source_label} sample_count mismatch.")
    if metadata.get("dataset_split") != expected_split:
        raise ValueError(f"{source_label} dataset split mismatch.")
    if metadata.get("checkpoint_path") != expected_checkpoint_path:
        raise ValueError(f"{source_label} checkpoint path mismatch.")
    if metadata.get("dataset_checksum_matches") is not True:
        raise ValueError(f"{source_label} dataset checksum must be valid.")


def build_comparison_rows(
    *,
    fgsm_curated_dir: str | Path,
    pgd_curated_dir: str | Path,
    epsilon: float = DEFAULT_EPSILON,
    expected_sample_count: int = 10000,
    expected_backend: str = "cupy",
    expected_gpu_name: str = "NVIDIA GeForce RTX 2080 Ti",
    expected_split: str = "test",
    expected_checkpoint_path: str = "results/checkpoints/portfolio_baseline_best.npz",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fgsm_dir = Path(fgsm_curated_dir)
    pgd_dir = Path(pgd_curated_dir)
    fgsm_metadata = load_json(fgsm_dir / "run_metadata.json")
    pgd_metadata = load_json(pgd_dir / "run_metadata.json")
    _validate_metadata(
        fgsm_metadata,
        expected_backend=expected_backend,
        expected_gpu_name=expected_gpu_name,
        expected_sample_count=expected_sample_count,
        expected_split=expected_split,
        expected_checkpoint_path=expected_checkpoint_path,
        source_label="FGSM",
    )
    _validate_metadata(
        pgd_metadata,
        expected_backend=expected_backend,
        expected_gpu_name=expected_gpu_name,
        expected_sample_count=expected_sample_count,
        expected_split=expected_split,
        expected_checkpoint_path=expected_checkpoint_path,
        source_label="PGD",
    )
    if not math.isclose(
        float(pgd_metadata["epsilon"]),
        epsilon,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("PGD metadata epsilon must match comparison epsilon.")

    fgsm_row = _row_for_epsilon(
        load_csv_rows(fgsm_dir / "robustness_summary.csv"),
        epsilon,
        source_label="FGSM",
    )
    pgd_rows = load_csv_rows(pgd_dir / "robustness_summary.csv")
    if len(pgd_rows) != 1:
        raise ValueError("PGD robustness summary must contain exactly one row.")
    pgd_row = pgd_rows[0]
    if not math.isclose(
        float(pgd_row["epsilon"]),
        epsilon,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("PGD row epsilon must match comparison epsilon.")

    fgsm_metrics = _validate_metric_row(
        fgsm_row,
        expected_sample_count=expected_sample_count,
        source_label="FGSM",
    )
    pgd_metrics = _validate_metric_row(
        pgd_row,
        expected_sample_count=expected_sample_count,
        source_label="PGD",
    )

    label = epsilon_label(epsilon)
    rows = [
        {
            "attack": "fgsm",
            "run_id": fgsm_metadata["run_id"],
            "epsilon": epsilon,
            "epsilon_label": label,
            "sample_count": expected_sample_count,
            "source_dir": str(fgsm_dir),
            **fgsm_metrics,
        },
        {
            "attack": "pgd_linf",
            "run_id": pgd_metadata["run_id"],
            "epsilon": epsilon,
            "epsilon_label": label,
            "sample_count": expected_sample_count,
            "source_dir": str(pgd_dir),
            **pgd_metrics,
        },
    ]
    metadata = {
        "schema_version": 1,
        "epsilon": epsilon,
        "epsilon_label": label,
        "sample_count": expected_sample_count,
        "backend": expected_backend,
        "gpu": expected_gpu_name,
        "dataset_split": expected_split,
        "checkpoint_path": expected_checkpoint_path,
        "fgsm_source": str(fgsm_dir),
        "pgd_source": str(pgd_dir),
        "fgsm_run_id": fgsm_metadata["run_id"],
        "pgd_run_id": pgd_metadata["run_id"],
        "interpretation": (
            "Empirical FGSM-vs-PGD comparison at a matched epsilon and "
            "dataset. Do not infer attack strength from labels alone; use the "
            "observed metrics."
        ),
        "results": rows,
    }
    return rows, metadata


def write_summary_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in SUMMARY_FIELDS} for row in rows)
    return output


def plot_fgsm_vs_pgd_summary(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = ["FGSM", "PGD-Linf"]
    clean_values = [float(row["clean_accuracy"]) for row in rows]
    adversarial_values = [float(row["adversarial_accuracy"]) for row in rows]
    asr_values = [float(row["attack_success_rate"]) for row in rows]
    x_positions = range(len(rows))

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    width = 0.35
    axes[0].bar(
        [x - width / 2 for x in x_positions],
        clean_values,
        width=width,
        label="Clean",
        color="#4c78a8",
    )
    axes[0].bar(
        [x + width / 2 for x in x_positions],
        adversarial_values,
        width=width,
        label="Adversarial",
        color="#f58518",
    )
    axes[0].set_xticks(list(x_positions), labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Clean vs adversarial accuracy")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, asr_values, color=["#e45756", "#72b7b2"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Attack success rate")
    axes[1].set_title("Observed attack success")
    axes[1].grid(axis="y", alpha=0.3)

    epsilon = rows[0]["epsilon_label"]
    sample_count = rows[0]["sample_count"]
    figure.suptitle(
        f"FGSM vs PGD-Linf | CIFAR-10 test | epsilon={epsilon} | "
        f"{sample_count} samples"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    existing = [output_dir / filename for filename in OUTPUT_FILES if (output_dir / filename).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Comparison output already exists. Pass --overwrite to regenerate: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_fgsm_pgd_comparison(
    *,
    fgsm_curated_dir: str | Path = DEFAULT_FGSM_CURATED_DIR,
    pgd_curated_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    epsilon: float = DEFAULT_EPSILON,
    expected_sample_count: int = 10000,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_path = _prepare_output_dir(Path(output_dir), overwrite=overwrite)
    rows, metadata = build_comparison_rows(
        fgsm_curated_dir=fgsm_curated_dir,
        pgd_curated_dir=pgd_curated_dir,
        epsilon=epsilon,
        expected_sample_count=expected_sample_count,
    )
    csv_path = write_summary_csv(rows, output_path / "fgsm_vs_pgd_summary.csv")
    json_path = save_metrics(metadata, output_path / "fgsm_vs_pgd_summary.json")
    plot_path = plot_fgsm_vs_pgd_summary(
        rows,
        output_path / "final_fgsm_vs_pgd_summary.png",
    )
    return {
        "fgsm_vs_pgd_summary_csv": csv_path,
        "fgsm_vs_pgd_summary_json": json_path,
        "final_fgsm_vs_pgd_summary": plot_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate curated FGSM-vs-PGD comparison artifacts from tracked "
            "curated run outputs."
        )
    )
    parser.add_argument(
        "--fgsm-curated-dir",
        type=Path,
        default=DEFAULT_FGSM_CURATED_DIR,
        help="Curated EWP3-E FGSM artifact directory.",
    )
    parser.add_argument(
        "--pgd-curated-dir",
        type=Path,
        required=True,
        help="Curated EWP4-D PGD artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for portfolio comparison artifacts.",
    )
    parser.add_argument(
        "--epsilon",
        default="8/255",
        help="Matched comparison epsilon as a float or fraction.",
    )
    parser.add_argument(
        "--expected-sample-count",
        type=int,
        default=10000,
        help="Required sample count for both curated sources.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate comparison artifacts if they already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_fgsm_pgd_comparison(
        fgsm_curated_dir=args.fgsm_curated_dir,
        pgd_curated_dir=args.pgd_curated_dir,
        output_dir=args.output_dir,
        epsilon=parse_single_epsilon(args.epsilon, name="epsilon"),
        expected_sample_count=args.expected_sample_count,
        overwrite=args.overwrite,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
