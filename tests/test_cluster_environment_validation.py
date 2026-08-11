import importlib.util
import json
from pathlib import Path

from configs.default_config import CIFAR10_ARCHIVE_NAME, CIFAR10_MD5


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "validate_cluster_environment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_cluster_environment",
    SCRIPT_PATH,
)
cluster_validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cluster_validation)


def test_collect_environment_numpy_does_not_require_cupy_or_gpu() -> None:
    report = cluster_validation.collect_environment("numpy")

    assert report["status"] == "passed"
    assert report["requested_backend"] == "numpy"
    assert report["backend_available"] is True
    assert report["python_version"]
    assert report["python_executable"]
    assert report["numpy_version"]
    assert "cupy" in report


def test_validate_dataset_reports_missing_archive_and_extracted_dir(
    tmp_path: Path,
) -> None:
    report = cluster_validation.validate_dataset(tmp_path)

    assert report["status"] == "failed"
    assert report["archive_exists"] is False
    assert report["archive_observed_md5"] is None
    assert report["archive_checksum_matches"] is False
    assert report["extracted_exists"] is False
    assert any("archive is missing" in error for error in report["errors"])
    assert any("extracted directory is missing" in error for error in report["errors"])


def test_validate_dataset_rejects_bad_archive_checksum(tmp_path: Path) -> None:
    archive_path = tmp_path / CIFAR10_ARCHIVE_NAME
    archive_path.write_bytes(b"not the official CIFAR-10 archive")

    report = cluster_validation.validate_dataset(tmp_path)

    assert report["status"] == "failed"
    assert report["archive_exists"] is True
    assert report["archive_observed_md5"] != CIFAR10_MD5
    assert report["archive_checksum_matches"] is False
    assert any("checksum mismatch" in error for error in report["errors"])


def test_main_writes_json_report_and_returns_nonzero_for_invalid_dataset(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    json_output = tmp_path / "reports" / "cluster_validation.json"

    exit_code = cluster_validation.main(
        [
            "--data-dir",
            str(data_dir),
            "--json-output",
            str(json_output),
        ]
    )

    assert exit_code == 1
    assert json_output.is_file()
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["environment"]["status"] == "passed"
    assert payload["dataset"]["status"] == "failed"
    assert payload["dataset"]["archive_path"].endswith(CIFAR10_ARCHIVE_NAME)
