import csv
import json

from experiments.generate_gradient_check_comparison import (
    RELATIVE_ERROR_THRESHOLD,
    build_gradient_check_results,
    generate_gradient_check_comparison,
)


def test_build_gradient_check_results_pass_existing_threshold() -> None:
    results = build_gradient_check_results()

    assert [(result.component, result.gradient_name) for result in results] == [
        ("Linear", "input"),
        ("Linear", "weights"),
        ("Linear", "bias"),
        ("Conv2D", "input"),
        ("Conv2D", "weights"),
        ("Conv2D", "bias"),
        ("SoftmaxCrossEntropyLoss", "logits"),
    ]
    assert max(result.max_relative_error for result in results) < (
        RELATIVE_ERROR_THRESHOLD
    )


def test_generate_gradient_check_comparison_writes_artifacts(tmp_path) -> None:
    paths = generate_gradient_check_comparison(tmp_path)

    assert paths["figure"] == tmp_path / "gradient_check_comparison.png"
    assert paths["csv"] == tmp_path / "gradient_check_comparison.csv"
    assert paths["summary"] == tmp_path / "gradient_check_comparison_summary.json"

    assert paths["figure"].is_file()
    assert paths["figure"].stat().st_size > 0
    assert paths["figure"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8")))
    assert len(rows) == 34
    assert set(rows[0]) == {
        "component",
        "gradient",
        "flat_index",
        "epsilon",
        "analytical",
        "numerical",
        "absolute_error",
        "relative_error",
    }

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["method"] == "centered finite difference"
    assert summary["total_elements"] == 34
    assert summary["max_relative_error"] < RELATIVE_ERROR_THRESHOLD
