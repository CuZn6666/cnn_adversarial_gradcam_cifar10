import csv
import json
from pathlib import Path

import pytest

from experiments.pgd import compare_fgsm_pgd


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_fgsm_curated(path: Path, *, checkpoint: str | None = None) -> None:
    checkpoint_path = checkpoint or "results/checkpoints/portfolio_baseline_best.npz"
    rows = [
        {
            "epsilon": 0.0,
            "epsilon_label": "0",
            "total_samples": 10000,
            "clean_correct": 4639,
            "adversarial_correct": 4639,
            "clean_correct_samples": 4639,
            "successful_attacks": 0,
            "clean_accuracy": 0.4639,
            "adversarial_accuracy": 0.4639,
            "accuracy_drop": 0.0,
            "attack_success_rate": 0.0,
        },
        {
            "epsilon": 8.0 / 255.0,
            "epsilon_label": "8/255",
            "total_samples": 10000,
            "clean_correct": 4639,
            "adversarial_correct": 99,
            "clean_correct_samples": 4639,
            "successful_attacks": 4540,
            "clean_accuracy": 0.4639,
            "adversarial_accuracy": 0.0099,
            "accuracy_drop": 0.45399999999999996,
            "attack_success_rate": 0.9786591937917655,
        },
    ]
    _write_csv(path / "robustness_summary.csv", rows)
    _write_json(
        path / "run_metadata.json",
        {
            "schema_version": 1,
            "run_id": "fgsm-run",
            "backend": "cupy",
            "gpu": "NVIDIA GeForce RTX 2080 Ti",
            "dataset_checksum_matches": True,
            "dataset_split": "test",
            "checkpoint_path": checkpoint_path,
            "sample_count": 10000,
            "batch_size": 128,
            "seed": 42,
        },
    )


def _write_pgd_curated(path: Path, *, sample_count: int = 10000) -> None:
    row = {
        "epsilon": 8.0 / 255.0,
        "alpha": 2.0 / 255.0,
        "steps": 10,
        "random_start": True,
        "total_samples": sample_count,
        "clean_correct": 4639,
        "adversarial_correct": 150,
        "clean_correct_samples": 4639,
        "successful_attacks": 4489,
        "clean_accuracy": 0.4639,
        "adversarial_accuracy": 0.015,
        "accuracy_drop": 0.4489,
        "attack_success_rate": 4489 / 4639,
    }
    _write_csv(path / "robustness_summary.csv", [row])
    _write_json(
        path / "run_metadata.json",
        {
            "schema_version": 1,
            "run_id": "pgd-run",
            "attack": "pgd_linf",
            "backend": "cupy",
            "gpu": "NVIDIA GeForce RTX 2080 Ti",
            "dataset_checksum_matches": True,
            "dataset_split": "test",
            "checkpoint_path": "results/checkpoints/portfolio_baseline_best.npz",
            "sample_count": sample_count,
            "batch_size": 128,
            "epsilon": 8.0 / 255.0,
            "alpha": 2.0 / 255.0,
            "steps": 10,
            "random_start": True,
            "seed": 42,
        },
    )


def test_build_fgsm_pgd_comparison_writes_structured_outputs(
    tmp_path: Path,
) -> None:
    fgsm_dir = tmp_path / "fgsm"
    pgd_dir = tmp_path / "pgd"
    output_dir = tmp_path / "portfolio"
    _write_fgsm_curated(fgsm_dir)
    _write_pgd_curated(pgd_dir)

    outputs = compare_fgsm_pgd.build_fgsm_pgd_comparison(
        fgsm_curated_dir=fgsm_dir,
        pgd_curated_dir=pgd_dir,
        output_dir=output_dir,
    )

    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    assert outputs["final_fgsm_vs_pgd_summary"].read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )

    with outputs["fgsm_vs_pgd_summary_csv"].open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["attack"] for row in rows] == ["fgsm", "pgd_linf"]
    assert rows[0]["adversarial_accuracy"] == "0.0099"
    assert rows[1]["sample_count"] == "10000"

    payload = json.loads(
        outputs["fgsm_vs_pgd_summary_json"].read_text(encoding="utf-8")
    )
    assert payload["epsilon_label"] == "8/255"
    assert payload["fgsm_run_id"] == "fgsm-run"
    assert payload["pgd_run_id"] == "pgd-run"
    assert "Do not infer attack strength" in payload["interpretation"]


def test_build_fgsm_pgd_comparison_rejects_unmatched_sample_count(
    tmp_path: Path,
) -> None:
    fgsm_dir = tmp_path / "fgsm"
    pgd_dir = tmp_path / "pgd"
    _write_fgsm_curated(fgsm_dir)
    _write_pgd_curated(pgd_dir, sample_count=32)

    with pytest.raises(ValueError, match="sample_count"):
        compare_fgsm_pgd.build_comparison_rows(
            fgsm_curated_dir=fgsm_dir,
            pgd_curated_dir=pgd_dir,
        )


def test_build_fgsm_pgd_comparison_rejects_checkpoint_mismatch(
    tmp_path: Path,
) -> None:
    fgsm_dir = tmp_path / "fgsm"
    pgd_dir = tmp_path / "pgd"
    _write_fgsm_curated(fgsm_dir, checkpoint="different.npz")
    _write_pgd_curated(pgd_dir)

    with pytest.raises(ValueError, match="checkpoint"):
        compare_fgsm_pgd.build_comparison_rows(
            fgsm_curated_dir=fgsm_dir,
            pgd_curated_dir=pgd_dir,
        )


def test_build_fgsm_pgd_comparison_requires_overwrite(tmp_path: Path) -> None:
    fgsm_dir = tmp_path / "fgsm"
    pgd_dir = tmp_path / "pgd"
    output_dir = tmp_path / "portfolio"
    _write_fgsm_curated(fgsm_dir)
    _write_pgd_curated(pgd_dir)

    compare_fgsm_pgd.build_fgsm_pgd_comparison(
        fgsm_curated_dir=fgsm_dir,
        pgd_curated_dir=pgd_dir,
        output_dir=output_dir,
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        compare_fgsm_pgd.build_fgsm_pgd_comparison(
            fgsm_curated_dir=fgsm_dir,
            pgd_curated_dir=pgd_dir,
            output_dir=output_dir,
        )
