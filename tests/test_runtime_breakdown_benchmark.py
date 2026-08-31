import csv

import pytest

from experiments.runtime.benchmark_runtime_breakdown import (
    DEFAULT_HISTORICAL_SOURCE,
    Conv2DBackwardOptimizationResult,
    RuntimeResult,
    benchmark_runtime_breakdown,
    build_operation_specs,
    generate_runtime_breakdown_artifacts,
    load_historical_conv2d_backward_optimization,
    plot_conv2d_backward_optimization,
    plot_runtime_breakdown,
    save_runtime_breakdown_csv,
    sort_runtime_results,
)


def test_build_operation_specs_covers_core_compact_cnn_operations() -> None:
    specs = build_operation_specs(batch_size=2, seed=42)

    assert [spec.operation for spec in specs] == [
        "Conv2D.forward",
        "Conv2D.backward",
        "MaxPool2D.forward",
        "MaxPool2D.backward",
        "Linear.forward",
        "Linear.backward",
        "ReLU.forward",
        "ReLU.backward",
    ]
    assert "(2, 3, 32, 32)" in specs[0].shape_summary
    assert "(2, 1024)" in specs[4].shape_summary

    for spec in specs:
        spec.setup()
        result = spec.run()
        assert result is not None


def test_sort_runtime_results_orders_slowest_first() -> None:
    results = [
        RuntimeResult("Linear.forward", 0.2, "shape"),
        RuntimeResult("Conv2D.backward", 3.1, "shape"),
        RuntimeResult("ReLU.forward", 0.01, "shape"),
    ]

    sorted_results = sort_runtime_results(results)

    assert [result.operation for result in sorted_results] == [
        "Conv2D.backward",
        "Linear.forward",
        "ReLU.forward",
    ]


def test_save_runtime_breakdown_csv_writes_operation_and_runtime(tmp_path) -> None:
    results = [
        RuntimeResult("Conv2D.backward", 1.23456789, "shape"),
        RuntimeResult("ReLU.forward", 0.001234, "shape"),
    ]
    output_path = tmp_path / "runtime_breakdown.csv"

    saved_path = save_runtime_breakdown_csv(results, output_path)

    assert saved_path == output_path
    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert rows == [
        {"operation": "Conv2D.backward", "runtime_ms": "1.234568"},
        {"operation": "ReLU.forward", "runtime_ms": "0.001234"},
    ]


def test_plot_runtime_breakdown_creates_png_and_pdf(tmp_path) -> None:
    results = [
        RuntimeResult("Conv2D.backward", 1.5, "shape"),
        RuntimeResult("Conv2D.forward", 0.5, "shape"),
        RuntimeResult("ReLU.forward", 0.01, "shape"),
    ]

    png_path, pdf_path = plot_runtime_breakdown(
        results,
        tmp_path / "runtime_breakdown.png",
        tmp_path / "runtime_breakdown.pdf",
    )

    assert png_path.is_file()
    assert pdf_path.is_file()
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_historical_conv2d_backward_optimization_is_traceable() -> None:
    result = load_historical_conv2d_backward_optimization(DEFAULT_HISTORICAL_SOURCE)

    assert result is not None
    assert result.before_seconds == pytest.approx(0.043458736)
    assert result.after_seconds == pytest.approx(0.000209222)
    assert result.speedup == pytest.approx(207.7159, rel=1e-5)


def test_plot_conv2d_backward_optimization_creates_png_and_pdf(tmp_path) -> None:
    result = Conv2DBackwardOptimizationResult(
        before_seconds=0.043458736,
        after_seconds=0.000209222,
        source_path=DEFAULT_HISTORICAL_SOURCE,
    )

    png_path, pdf_path = plot_conv2d_backward_optimization(
        result,
        tmp_path / "conv2d_backward_optimization.png",
        tmp_path / "conv2d_backward_optimization.pdf",
    )

    assert png_path.is_file()
    assert pdf_path.is_file()
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_benchmark_runtime_breakdown_smoke_uses_positive_runtimes() -> None:
    results = benchmark_runtime_breakdown(
        batch_size=1,
        seed=42,
        warmup_iterations=0,
        repetitions=1,
    )

    assert len(results) == 8
    assert all(result.runtime_ms > 0.0 for result in results)
    assert results == sort_runtime_results(results)


def test_generate_runtime_breakdown_artifacts_writes_expected_files(tmp_path) -> None:
    artifacts = generate_runtime_breakdown_artifacts(
        figure_dir=tmp_path / "figures",
        csv_path=tmp_path / "runtime_breakdown.csv",
        batch_size=1,
        seed=42,
        warmup_iterations=0,
        repetitions=1,
        historical_source=DEFAULT_HISTORICAL_SOURCE,
    )

    assert artifacts["runtime_breakdown_png"].is_file()
    assert artifacts["runtime_breakdown_pdf"].is_file()
    assert artifacts["runtime_breakdown_csv"].is_file()
    assert artifacts["conv2d_optimization"] is not None
    assert artifacts["conv2d_optimization_png"].is_file()
    assert artifacts["conv2d_optimization_pdf"].is_file()
