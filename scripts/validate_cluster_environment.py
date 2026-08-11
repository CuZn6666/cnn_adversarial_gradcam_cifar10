from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.util
import json
import platform
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.default_config import (
    CIFAR10_ARCHIVE_NAME,
    CIFAR10_EXTRACTED_DIR,
    CIFAR10_MD5,
    DATA_DIR,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
)
from src.data.cifar10_loader import compute_md5, load_batch, load_label_names


EXPECTED_TRAIN_SHAPE = (50000, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
EXPECTED_TEST_SHAPE = (10000, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _decode_device_name(raw_name: Any) -> str:
    if isinstance(raw_name, bytes):
        return raw_name.decode("utf-8", errors="replace")
    return str(raw_name)


def collect_environment(requested_backend: str) -> dict[str, Any]:
    """Collect environment metadata without requiring a GPU for NumPy checks."""
    errors: list[str] = []
    cupy_info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_runtime_version": None,
        "device_count": None,
        "gpu_name": None,
        "error": None,
    }

    if importlib.util.find_spec("cupy") is not None:
        cupy_info["installed"] = True
        try:
            import cupy as cp

            cupy_info["version"] = str(cp.__version__)
            cupy_info["cuda_runtime_version"] = int(
                cp.cuda.runtime.runtimeGetVersion()
            )
            device_count = int(cp.cuda.runtime.getDeviceCount())
            cupy_info["device_count"] = device_count
            if device_count > 0:
                properties = cp.cuda.runtime.getDeviceProperties(0)
                cupy_info["gpu_name"] = _decode_device_name(
                    properties.get("name", "unknown")
                )
        except Exception as error:  # pragma: no cover - hardware dependent.
            cupy_info["error"] = str(error)
    else:
        cupy_info["error"] = "cupy is not installed"

    if requested_backend == "cupy":
        if not cupy_info["installed"]:
            errors.append("CuPy backend requested, but cupy is not installed.")
        elif cupy_info["error"] is not None:
            errors.append(
                "CuPy backend requested, but CUDA/CuPy validation failed: "
                f"{cupy_info['error']}"
            )
        elif not cupy_info["device_count"]:
            errors.append("CuPy backend requested, but no CUDA GPU is visible.")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "requested_backend": requested_backend,
        "backend_available": not errors,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "cupy": cupy_info,
    }


def _extract_archive(archive_path: Path, data_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")


def validate_dataset(
    data_dir: Path,
    *,
    extract_if_needed: bool = False,
) -> dict[str, Any]:
    """Validate a staged CIFAR-10 archive and extracted Python dataset."""
    errors: list[str] = []
    data_dir = Path(data_dir)
    archive_path = data_dir / CIFAR10_ARCHIVE_NAME
    extracted_dir = data_dir / CIFAR10_EXTRACTED_DIR

    archive_md5 = None
    archive_size_bytes = None
    archive_exists = archive_path.is_file()
    if archive_exists:
        archive_size_bytes = archive_path.stat().st_size
        archive_md5 = compute_md5(archive_path)
        if archive_md5 != CIFAR10_MD5:
            errors.append(
                "CIFAR-10 archive checksum mismatch. "
                f"Expected {CIFAR10_MD5}, observed {archive_md5}."
            )
    else:
        errors.append(f"CIFAR-10 archive is missing: {archive_path}")

    extracted_before = extracted_dir.is_dir()
    extracted_performed = False
    if (
        extract_if_needed
        and archive_exists
        and archive_md5 == CIFAR10_MD5
        and not extracted_before
    ):
        try:
            _extract_archive(archive_path, data_dir)
            extracted_performed = True
        except Exception as error:
            errors.append(f"CIFAR-10 extraction failed: {error}")

    extracted_exists = extracted_dir.is_dir()
    if not extracted_exists:
        errors.append(f"CIFAR-10 extracted directory is missing: {extracted_dir}")

    split_report: dict[str, Any] = {
        "train_shape": None,
        "train_labels_shape": None,
        "test_shape": None,
        "test_labels_shape": None,
        "class_count": None,
    }
    if extracted_exists:
        try:
            train_batches = [
                load_batch(extracted_dir / f"data_batch_{index}")
                for index in range(1, 6)
            ]
            x_train = np.concatenate(
                [batch_images for batch_images, _ in train_batches],
                axis=0,
            )
            y_train = np.concatenate(
                [batch_labels for _, batch_labels in train_batches],
                axis=0,
            )
            x_test, y_test = load_batch(extracted_dir / "test_batch")
            class_names = load_label_names(extracted_dir)

            split_report = {
                "train_shape": list(x_train.shape),
                "train_labels_shape": list(y_train.shape),
                "test_shape": list(x_test.shape),
                "test_labels_shape": list(y_test.shape),
                "class_count": len(class_names),
            }

            if x_train.shape != EXPECTED_TRAIN_SHAPE:
                errors.append(
                    f"Unexpected train image shape: {x_train.shape}; "
                    f"expected {EXPECTED_TRAIN_SHAPE}."
                )
            if y_train.shape != (EXPECTED_TRAIN_SHAPE[0],):
                errors.append(
                    f"Unexpected train label shape: {y_train.shape}; "
                    f"expected {(EXPECTED_TRAIN_SHAPE[0],)}."
                )
            if x_test.shape != EXPECTED_TEST_SHAPE:
                errors.append(
                    f"Unexpected test image shape: {x_test.shape}; "
                    f"expected {EXPECTED_TEST_SHAPE}."
                )
            if y_test.shape != (EXPECTED_TEST_SHAPE[0],):
                errors.append(
                    f"Unexpected test label shape: {y_test.shape}; "
                    f"expected {(EXPECTED_TEST_SHAPE[0],)}."
                )
            if len(class_names) != NUM_CLASSES:
                errors.append(
                    f"Unexpected class count: {len(class_names)}; "
                    f"expected {NUM_CLASSES}."
                )
        except Exception as error:
            errors.append(f"CIFAR-10 split loading failed: {error}")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "data_dir": str(data_dir),
        "archive_path": str(archive_path),
        "archive_expected_md5": CIFAR10_MD5,
        "archive_observed_md5": archive_md5,
        "archive_size_bytes": archive_size_bytes,
        "archive_exists": archive_exists,
        "archive_checksum_matches": archive_md5 == CIFAR10_MD5,
        "extracted_dir": str(extracted_dir),
        "extracted_exists": extracted_exists,
        "extracted_performed": extracted_performed,
        **split_report,
    }


def build_report(
    *,
    data_dir: Path,
    requested_backend: str,
    extract_if_needed: bool,
) -> dict[str, Any]:
    environment = collect_environment(requested_backend)
    dataset = validate_dataset(
        data_dir,
        extract_if_needed=extract_if_needed,
    )
    status = (
        "passed"
        if environment["status"] == "passed" and dataset["status"] == "passed"
        else "failed"
    )
    return {
        "run_id": _utc_timestamp(),
        "status": status,
        "environment": environment,
        "dataset": dataset,
    }


def print_human_report(report: dict[str, Any]) -> None:
    environment = report["environment"]
    dataset = report["dataset"]
    cupy = environment["cupy"]

    print("Cluster environment and CIFAR-10 staging validation")
    print(f"Status: {report['status'].upper()}")
    print(f"Run ID: {report['run_id']}")
    print()
    print("Environment")
    print(f"  requested_backend: {environment['requested_backend']}")
    print(f"  backend_available: {environment['backend_available']}")
    print(f"  python_version: {environment['python_version']}")
    print(f"  python_executable: {environment['python_executable']}")
    print(f"  numpy_version: {environment['numpy_version']}")
    print(f"  cupy_installed: {cupy['installed']}")
    print(f"  cupy_version: {cupy['version']}")
    print(f"  cuda_runtime_version: {cupy['cuda_runtime_version']}")
    print(f"  cuda_device_count: {cupy['device_count']}")
    print(f"  gpu_name: {cupy['gpu_name']}")
    if cupy["error"]:
        print(f"  cupy_error: {cupy['error']}")
    print()
    print("Dataset")
    print(f"  data_dir: {dataset['data_dir']}")
    print(f"  archive_path: {dataset['archive_path']}")
    print(f"  archive_exists: {dataset['archive_exists']}")
    print(f"  archive_size_bytes: {dataset['archive_size_bytes']}")
    print(f"  archive_expected_md5: {dataset['archive_expected_md5']}")
    print(f"  archive_observed_md5: {dataset['archive_observed_md5']}")
    print(f"  archive_checksum_matches: {dataset['archive_checksum_matches']}")
    print(f"  extracted_dir: {dataset['extracted_dir']}")
    print(f"  extracted_exists: {dataset['extracted_exists']}")
    print(f"  extracted_performed: {dataset['extracted_performed']}")
    print(f"  train_shape: {dataset['train_shape']}")
    print(f"  train_labels_shape: {dataset['train_labels_shape']}")
    print(f"  test_shape: {dataset['test_shape']}")
    print(f"  test_labels_shape: {dataset['test_labels_shape']}")
    print(f"  class_count: {dataset['class_count']}")

    errors = environment["errors"] + dataset["errors"]
    if errors:
        print()
        print("Errors")
        for error in errors:
            print(f"  - {error}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Python/CuPy environment and staged CIFAR-10 data "
            "without running training or FGSM experiments."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the CIFAR-10 archive and extracted data.",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "cupy"),
        default="numpy",
        help="Backend that must be available for this validation run.",
    )
    parser.add_argument(
        "--extract-if-needed",
        action="store_true",
        help=(
            "Extract CIFAR-10 only when the archive is present, checksum-valid, "
            "and the extracted directory is missing."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON validation report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        data_dir=args.data_dir,
        requested_backend=args.backend,
        extract_if_needed=args.extract_if_needed,
    )
    print_human_report(report)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
