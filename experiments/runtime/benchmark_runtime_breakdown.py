"""Runtime breakdown benchmark for core CompactCNN operations."""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.default_config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SEED
from src.models import CompactCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_SIZE = 32
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_REPETITIONS = 25
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_CSV_PATH = PROJECT_ROOT / "results" / "runtime_breakdown.csv"
DEFAULT_HISTORICAL_SOURCE = (
    PROJECT_ROOT / "deliverables" / "WP6" / "runtime_benchmark_after.md"
)
RUNTIME_BREAKDOWN_TITLE = "Runtime Breakdown of Core CNN Operations"
HISTORICAL_CONV2D_BACKWARD_PATTERN = re.compile(
    r"Conv2D\.backward:\s+"
    r"(?P<before>[0-9]*\.?[0-9]+)\s+->\s+"
    r"(?P<after>[0-9]*\.?[0-9]+)\s+seconds"
)
HISTORICAL_CONV2D_BACKWARD_VALUE_PATTERN = re.compile(
    r"Conv2D\.backward:\s+"
    r"(?P<value>[0-9]*\.?[0-9]+)\s+seconds per iteration"
)


@dataclass(frozen=True)
class OperationSpec:
    operation: str
    setup: Callable[[], None]
    run: Callable[[], object]
    shape_summary: str


@dataclass(frozen=True)
class RuntimeResult:
    operation: str
    runtime_ms: float
    shape_summary: str


@dataclass(frozen=True)
class Conv2DBackwardOptimizationResult:
    before_seconds: float
    after_seconds: float
    source_path: Path

    @property
    def before_ms(self) -> float:
        return self.before_seconds * 1000.0

    @property
    def after_ms(self) -> float:
        return self.after_seconds * 1000.0

    @property
    def speedup(self) -> float:
        return self.before_seconds / self.after_seconds


def _random_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    return rng.normal(size=shape).astype(np.float32)


def build_operation_specs(
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = SEED,
) -> list[OperationSpec]:
    """Build benchmark operations from actual CompactCNN layer instances."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    rng = np.random.default_rng(seed)
    model = CompactCNN(seed=seed, backend="numpy")
    images = rng.random(
        (batch_size, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.float32,
    )

    conv1_output = model.conv1.forward(images)
    relu1_output = model.relu1.forward(conv1_output)
    pool1_output = model.pool1.forward(relu1_output)
    conv2_output = model.conv2.forward(pool1_output)
    relu2_output = model.relu2.forward(conv2_output)
    pool2_output = model.pool2.forward(relu2_output)
    flattened = model.flatten.forward(pool2_output)
    logits = model.classifier.forward(flattened)

    conv1_grad_out = _random_normal(rng, conv1_output.shape)
    conv2_grad_out = _random_normal(rng, conv2_output.shape)
    relu1_grad_out = _random_normal(rng, relu1_output.shape)
    relu2_grad_out = _random_normal(rng, relu2_output.shape)
    pool1_grad_out = _random_normal(rng, pool1_output.shape)
    pool2_grad_out = _random_normal(rng, pool2_output.shape)
    logits_grad_out = _random_normal(rng, logits.shape)

    def setup_conv2d_backward() -> None:
        model.conv1.forward(images)
        model.conv2.forward(pool1_output)

    def setup_relu_backward() -> None:
        model.relu1.forward(conv1_output)
        model.relu2.forward(conv2_output)

    def setup_max_pool2d_backward() -> None:
        model.pool1.forward(relu1_output)
        model.pool2.forward(relu2_output)

    def setup_linear_backward() -> None:
        model.classifier.forward(flattened)

    return [
        OperationSpec(
            "Conv2D.forward",
            setup=lambda: None,
            run=lambda: (
                model.conv1.forward(images),
                model.conv2.forward(pool1_output),
            ),
            shape_summary=(
                f"{images.shape}->{conv1_output.shape}; "
                f"{pool1_output.shape}->{conv2_output.shape}"
            ),
        ),
        OperationSpec(
            "Conv2D.backward",
            setup=setup_conv2d_backward,
            run=lambda: (
                model.conv2.backward(conv2_grad_out),
                model.conv1.backward(conv1_grad_out),
            ),
            shape_summary=(
                f"{conv2_grad_out.shape}->{pool1_output.shape}; "
                f"{conv1_grad_out.shape}->{images.shape}"
            ),
        ),
        OperationSpec(
            "MaxPool2D.forward",
            setup=lambda: None,
            run=lambda: (
                model.pool1.forward(relu1_output),
                model.pool2.forward(relu2_output),
            ),
            shape_summary=(
                f"{relu1_output.shape}->{pool1_output.shape}; "
                f"{relu2_output.shape}->{pool2_output.shape}"
            ),
        ),
        OperationSpec(
            "MaxPool2D.backward",
            setup=setup_max_pool2d_backward,
            run=lambda: (
                model.pool2.backward(pool2_grad_out),
                model.pool1.backward(pool1_grad_out),
            ),
            shape_summary=(
                f"{pool2_grad_out.shape}->{relu2_output.shape}; "
                f"{pool1_grad_out.shape}->{relu1_output.shape}"
            ),
        ),
        OperationSpec(
            "Linear.forward",
            setup=lambda: None,
            run=lambda: model.classifier.forward(flattened),
            shape_summary=f"{flattened.shape}->{logits.shape}",
        ),
        OperationSpec(
            "Linear.backward",
            setup=setup_linear_backward,
            run=lambda: model.classifier.backward(logits_grad_out),
            shape_summary=f"{logits_grad_out.shape}->{flattened.shape}",
        ),
        OperationSpec(
            "ReLU.forward",
            setup=lambda: None,
            run=lambda: (
                model.relu1.forward(conv1_output),
                model.relu2.forward(conv2_output),
            ),
            shape_summary=(
                f"{conv1_output.shape}->{relu1_output.shape}; "
                f"{conv2_output.shape}->{relu2_output.shape}"
            ),
        ),
        OperationSpec(
            "ReLU.backward",
            setup=setup_relu_backward,
            run=lambda: (
                model.relu2.backward(relu2_grad_out),
                model.relu1.backward(relu1_grad_out),
            ),
            shape_summary=(
                f"{relu2_grad_out.shape}->{conv2_output.shape}; "
                f"{relu1_grad_out.shape}->{conv1_output.shape}"
            ),
        ),
    ]


def measure_operation_runtime_ms(
    spec: OperationSpec,
    warmup_iterations: int,
    repetitions: int,
) -> float:
    """Measure median runtime for one operation in milliseconds."""
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    for _ in range(warmup_iterations):
        spec.setup()
        spec.run()

    elapsed_seconds = []
    for _ in range(repetitions):
        spec.setup()
        start = perf_counter()
        spec.run()
        elapsed_seconds.append(perf_counter() - start)

    return float(np.median(elapsed_seconds) * 1000.0)


def benchmark_runtime_breakdown(
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = SEED,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    repetitions: int = DEFAULT_REPETITIONS,
) -> list[RuntimeResult]:
    specs = build_operation_specs(batch_size=batch_size, seed=seed)
    results = [
        RuntimeResult(
            operation=spec.operation,
            runtime_ms=measure_operation_runtime_ms(
                spec,
                warmup_iterations=warmup_iterations,
                repetitions=repetitions,
            ),
            shape_summary=spec.shape_summary,
        )
        for spec in specs
    ]
    return sort_runtime_results(results)


def sort_runtime_results(results: Sequence[RuntimeResult]) -> list[RuntimeResult]:
    return sorted(results, key=lambda result: result.runtime_ms, reverse=True)


def save_runtime_breakdown_csv(
    results: Sequence[RuntimeResult],
    output_path: str | Path = DEFAULT_CSV_PATH,
) -> Path:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["operation", "runtime_ms"],
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "operation": result.operation,
                    "runtime_ms": f"{result.runtime_ms:.6f}",
                },
            )
    return csv_path


def plot_runtime_breakdown(
    results: Sequence[RuntimeResult],
    png_path: str | Path,
    pdf_path: str | Path,
) -> tuple[Path, Path]:
    if not results:
        raise ValueError("Runtime results must not be empty.")

    sorted_results = sort_runtime_results(results)
    labels = [result.operation for result in sorted_results]
    runtimes = [result.runtime_ms for result in sorted_results]

    output_png = Path(png_path)
    output_pdf = Path(pdf_path)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    axes.set_facecolor("white")
    bars = axes.barh(
        labels,
        runtimes,
        color="#4C78A8",
        edgecolor="#2F4B66",
        linewidth=0.8,
        zorder=3,
    )
    axes.invert_yaxis()
    axes.set_title(RUNTIME_BREAKDOWN_TITLE, pad=12)
    axes.set_xlabel("Runtime (ms)")
    axes.set_ylabel("Operation")
    axes.grid(True, axis="x", alpha=0.25, linewidth=0.7, zorder=0)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    value_padding = max(runtimes) * 0.012 if max(runtimes) > 0 else 0.001
    axes.set_xlim(0.0, max(runtimes) * 1.16)
    for bar, runtime_ms in zip(bars, runtimes):
        axes.text(
            runtime_ms + value_padding,
            bar.get_y() + bar.get_height() / 2.0,
            f"{runtime_ms:.3f} ms",
            va="center",
            ha="left",
            fontsize=9,
        )

    figure.tight_layout()
    figure.savefig(output_png, dpi=300)
    figure.savefig(output_pdf)
    plt.close(figure)
    return output_png, output_pdf


def load_historical_conv2d_backward_optimization(
    source_path: str | Path = DEFAULT_HISTORICAL_SOURCE,
) -> Conv2DBackwardOptimizationResult | None:
    source = Path(source_path)
    if not source.exists():
        return None
    source_text = source.read_text(encoding="utf-8")
    match = HISTORICAL_CONV2D_BACKWARD_PATTERN.search(source_text)
    if match is not None:
        before_seconds = float(match.group("before"))
        after_seconds = float(match.group("after"))
    else:
        values = [
            float(value)
            for value in HISTORICAL_CONV2D_BACKWARD_VALUE_PATTERN.findall(
                source_text,
            )
        ]
        if len(values) < 2:
            return None
        before_seconds, after_seconds = values[0], values[1]

    if before_seconds <= 0.0 or after_seconds <= 0.0:
        return None
    return Conv2DBackwardOptimizationResult(
        before_seconds=before_seconds,
        after_seconds=after_seconds,
        source_path=source,
    )


def plot_conv2d_backward_optimization(
    result: Conv2DBackwardOptimizationResult,
    png_path: str | Path,
    pdf_path: str | Path,
) -> tuple[Path, Path]:
    output_png = Path(png_path)
    output_pdf = Path(pdf_path)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    labels = [
        "Before optimization",
        "After optimization",
    ]
    runtimes = [result.before_ms, result.after_ms]

    figure, axes = plt.subplots(figsize=(7.0, 4.4), facecolor="white")
    axes.set_facecolor("white")
    bars = axes.barh(
        labels,
        runtimes,
        color=["#D95F02", "#1B9E77"],
        edgecolor="#333333",
        linewidth=0.8,
        zorder=3,
    )
    axes.invert_yaxis()
    axes.set_xscale("log")
    axes.set_xlabel("Runtime (ms, log scale)")
    axes.set_title("Conv2D.backward Optimization Runtime", pad=12)
    axes.grid(True, axis="x", alpha=0.25, linewidth=0.7, which="both", zorder=0)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    for bar, runtime_ms in zip(bars, runtimes):
        axes.text(
            runtime_ms * 1.08,
            bar.get_y() + bar.get_height() / 2.0,
            f"{runtime_ms:.3f} ms",
            va="center",
            ha="left",
            fontsize=9,
        )

    axes.text(
        0.5,
        0.08,
        f"Speedup: {result.speedup:.2f}x",
        transform=axes.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    figure.tight_layout()
    figure.savefig(output_png, dpi=300)
    figure.savefig(output_pdf)
    plt.close(figure)
    return output_png, output_pdf


def generate_runtime_breakdown_artifacts(
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = SEED,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    repetitions: int = DEFAULT_REPETITIONS,
    historical_source: str | Path = DEFAULT_HISTORICAL_SOURCE,
) -> dict[str, Path | list[RuntimeResult] | Conv2DBackwardOptimizationResult | None]:
    results = benchmark_runtime_breakdown(
        batch_size=batch_size,
        seed=seed,
        warmup_iterations=warmup_iterations,
        repetitions=repetitions,
    )
    figure_root = Path(figure_dir)
    runtime_png, runtime_pdf = plot_runtime_breakdown(
        results,
        figure_root / "runtime_breakdown.png",
        figure_root / "runtime_breakdown.pdf",
    )
    csv_output = save_runtime_breakdown_csv(results, csv_path)

    optimization = load_historical_conv2d_backward_optimization(historical_source)
    optimization_png = None
    optimization_pdf = None
    if optimization is not None:
        optimization_png, optimization_pdf = plot_conv2d_backward_optimization(
            optimization,
            figure_root / "conv2d_backward_optimization.png",
            figure_root / "conv2d_backward_optimization.pdf",
        )

    return {
        "results": results,
        "runtime_breakdown_png": runtime_png,
        "runtime_breakdown_pdf": runtime_pdf,
        "runtime_breakdown_csv": csv_output,
        "conv2d_optimization": optimization,
        "conv2d_optimization_png": optimization_png,
        "conv2d_optimization_pdf": optimization_pdf,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and plot CompactCNN core operation runtimes.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS,
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
    )
    parser.add_argument(
        "--historical-source",
        type=Path,
        default=DEFAULT_HISTORICAL_SOURCE,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts = generate_runtime_breakdown_artifacts(
        figure_dir=args.figure_dir,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        seed=args.seed,
        warmup_iterations=args.warmup_iterations,
        repetitions=args.repetitions,
        historical_source=args.historical_source,
    )

    print("Runtime breakdown benchmark")
    print(f"batch_size={args.batch_size}")
    print(f"seed={args.seed}")
    print(f"warmup_iterations={args.warmup_iterations}")
    print(f"repetitions={args.repetitions}")
    for result in artifacts["results"]:
        print(f"{result.operation}: {result.runtime_ms:.6f} ms")

    optimization = artifacts["conv2d_optimization"]
    if optimization is None:
        print("Conv2D.backward optimization comparison skipped: no source data.")
    else:
        print(
            "Conv2D.backward optimization: "
            f"before={optimization.before_ms:.6f} ms, "
            f"after={optimization.after_ms:.6f} ms, "
            f"speedup={optimization.speedup:.2f}x, "
            f"source={optimization.source_path}"
        )

    print(f"runtime_breakdown_png: {artifacts['runtime_breakdown_png']}")
    print(f"runtime_breakdown_pdf: {artifacts['runtime_breakdown_pdf']}")
    print(f"runtime_breakdown_csv: {artifacts['runtime_breakdown_csv']}")
    if artifacts["conv2d_optimization_png"] is not None:
        print(f"conv2d_optimization_png: {artifacts['conv2d_optimization_png']}")
    if artifacts["conv2d_optimization_pdf"] is not None:
        print(f"conv2d_optimization_pdf: {artifacts['conv2d_optimization_pdf']}")


if __name__ == "__main__":
    main()
