from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.layers import Conv2D, Linear
from src.losses import SoftmaxCrossEntropyLoss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "curated" / "portfolio"
RELATIVE_ERROR_THRESHOLD = 1e-4
DISPLAY_COMPONENT_NAMES = {
    "SoftmaxCrossEntropyLoss": "Softmax CE",
}


@dataclass(frozen=True)
class GradientCheckResult:
    component: str
    gradient_name: str
    analytical: np.ndarray
    numerical: np.ndarray
    epsilon: float

    @property
    def relative_errors(self) -> np.ndarray:
        denominator = np.maximum(
            1e-12,
            np.abs(self.analytical) + np.abs(self.numerical),
        )
        return np.abs(self.analytical - self.numerical) / denominator

    @property
    def max_relative_error(self) -> float:
        return float(np.max(self.relative_errors))

    @property
    def max_absolute_error(self) -> float:
        return float(np.max(np.abs(self.analytical - self.numerical)))

    @property
    def element_count(self) -> int:
        return int(self.analytical.size)


def centered_finite_difference(
    values: np.ndarray,
    objective: Callable[[], float],
    epsilon: float = 1e-6,
) -> np.ndarray:
    numerical_gradient = np.zeros_like(values)

    for index in np.ndindex(values.shape):
        original_value = values[index]

        values[index] = original_value + epsilon
        objective_plus = objective()

        values[index] = original_value - epsilon
        objective_minus = objective()

        values[index] = original_value
        numerical_gradient[index] = (
            objective_plus - objective_minus
        ) / (2.0 * epsilon)

    return numerical_gradient


def _make_result(
    component: str,
    gradient_name: str,
    analytical: np.ndarray,
    numerical: np.ndarray,
    epsilon: float,
) -> GradientCheckResult:
    analytical_array = np.asarray(analytical, dtype=np.float64)
    numerical_array = np.asarray(numerical, dtype=np.float64)
    if analytical_array.shape != numerical_array.shape:
        raise ValueError("Analytical and numerical gradients must have equal shapes.")
    if not np.isfinite(analytical_array).all() or not np.isfinite(
        numerical_array,
    ).all():
        raise ValueError("Gradients must contain only finite values.")
    return GradientCheckResult(
        component=component,
        gradient_name=gradient_name,
        analytical=analytical_array.copy(),
        numerical=numerical_array.copy(),
        epsilon=float(epsilon),
    )


def _linear_gradient_checks() -> list[GradientCheckResult]:
    epsilon = 1e-6
    rng = np.random.default_rng(42)
    layer = Linear(in_features=3, out_features=2, rng=rng)
    layer.weights = rng.normal(size=(2, 3))
    layer.bias = rng.normal(size=2)
    inputs = rng.normal(size=(2, 3))
    grad_out = rng.normal(size=(2, 2))

    layer.forward(inputs)
    analytical_grad_input = layer.backward(grad_out).copy()
    analytical_grad_weight = layer.grad_weight.copy()
    analytical_grad_bias = layer.grad_bias.copy()

    def objective() -> float:
        return float(np.sum(layer.forward(inputs) * grad_out))

    return [
        _make_result(
            "Linear",
            "input",
            analytical_grad_input,
            centered_finite_difference(inputs, objective, epsilon),
            epsilon,
        ),
        _make_result(
            "Linear",
            "weights",
            analytical_grad_weight,
            centered_finite_difference(layer.weights, objective, epsilon),
            epsilon,
        ),
        _make_result(
            "Linear",
            "bias",
            analytical_grad_bias,
            centered_finite_difference(layer.bias, objective, epsilon),
            epsilon,
        ),
    ]


def _conv2d_gradient_checks() -> list[GradientCheckResult]:
    epsilon = 1e-3
    rng = np.random.default_rng(7)
    layer = Conv2D(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        rng=rng,
    )
    layer.weights = rng.uniform(0.5, 1.5, size=(1, 1, 2, 2))
    layer.bias = rng.uniform(0.5, 1.5, size=1)
    inputs = rng.uniform(0.5, 1.5, size=(1, 1, 3, 3))
    grad_out = rng.uniform(0.5, 1.5, size=(1, 1, 2, 2))

    layer.forward(inputs)
    analytical_grad_input = layer.backward(grad_out).copy()
    analytical_grad_weight = layer.grad_weight.copy()
    analytical_grad_bias = layer.grad_bias.copy()

    def objective() -> float:
        return float(np.sum(layer.forward(inputs) * grad_out))

    return [
        _make_result(
            "Conv2D",
            "input",
            analytical_grad_input,
            centered_finite_difference(inputs, objective, epsilon),
            epsilon,
        ),
        _make_result(
            "Conv2D",
            "weights",
            analytical_grad_weight,
            centered_finite_difference(layer.weights, objective, epsilon),
            epsilon,
        ),
        _make_result(
            "Conv2D",
            "bias",
            analytical_grad_bias,
            centered_finite_difference(layer.bias, objective, epsilon),
            epsilon,
        ),
    ]


def _softmax_cross_entropy_gradient_checks() -> list[GradientCheckResult]:
    epsilon = 1e-6
    logits = np.array(
        [[0.2, -0.1, 0.4], [0.1, 0.3, -0.2]],
        dtype=np.float64,
    )
    labels = np.array([2, 0], dtype=np.int64)
    loss_function = SoftmaxCrossEntropyLoss()

    loss_function.forward(logits, labels)
    analytical_grad_logits = loss_function.backward()

    def objective() -> float:
        return loss_function.forward(logits, labels)

    return [
        _make_result(
            "SoftmaxCrossEntropyLoss",
            "logits",
            analytical_grad_logits,
            centered_finite_difference(logits, objective, epsilon),
            epsilon,
        ),
    ]


def build_gradient_check_results() -> list[GradientCheckResult]:
    return [
        *_linear_gradient_checks(),
        *_conv2d_gradient_checks(),
        *_softmax_cross_entropy_gradient_checks(),
    ]


def _gradient_summary(
    results: Sequence[GradientCheckResult],
    threshold: float,
) -> dict[str, object]:
    max_relative_error = max(result.max_relative_error for result in results)
    return {
        "status": "PASS" if max_relative_error < threshold else "FAIL",
        "method": "centered finite difference",
        "relative_error_threshold": float(threshold),
        "total_elements": int(sum(result.element_count for result in results)),
        "max_relative_error": float(max_relative_error),
        "checks": [
            {
                "component": result.component,
                "gradient": result.gradient_name,
                "shape": list(result.analytical.shape),
                "epsilon": result.epsilon,
                "element_count": result.element_count,
                "max_absolute_error": result.max_absolute_error,
                "max_relative_error": result.max_relative_error,
            }
            for result in results
        ],
    }


def write_gradient_check_csv(
    results: Sequence[GradientCheckResult],
    output_path: str | Path,
) -> Path:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "component",
        "gradient",
        "flat_index",
        "epsilon",
        "analytical",
        "numerical",
        "absolute_error",
        "relative_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            analytical = result.analytical.reshape(-1)
            numerical = result.numerical.reshape(-1)
            absolute_errors = np.abs(analytical - numerical)
            relative_errors = result.relative_errors.reshape(-1)
            for index, (
                analytical_value,
                numerical_value,
                absolute_error,
                relative_error,
            ) in enumerate(
                zip(
                    analytical,
                    numerical,
                    absolute_errors,
                    relative_errors,
                ),
            ):
                writer.writerow(
                    {
                        "component": result.component,
                        "gradient": result.gradient_name,
                        "flat_index": index,
                        "epsilon": f"{result.epsilon:.17g}",
                        "analytical": f"{float(analytical_value):.17g}",
                        "numerical": f"{float(numerical_value):.17g}",
                        "absolute_error": f"{float(absolute_error):.17g}",
                        "relative_error": f"{float(relative_error):.17g}",
                    },
                )
    return csv_path


def write_gradient_check_summary_json(
    results: Sequence[GradientCheckResult],
    output_path: str | Path,
    threshold: float = RELATIVE_ERROR_THRESHOLD,
) -> Path:
    summary_path = Path(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _gradient_summary(results, threshold)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def plot_gradient_check_comparison(
    results: Sequence[GradientCheckResult],
    output_path: str | Path,
    threshold: float = RELATIVE_ERROR_THRESHOLD,
) -> Path:
    if not results:
        raise ValueError("Gradient check results must not be empty.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be a positive finite number.")

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    analytical_values: list[float] = []
    numerical_values: list[float] = []
    relative_errors: list[float] = []
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    boundaries: list[float] = []

    offset = 0
    for result in results:
        analytical = result.analytical.reshape(-1)
        numerical = result.numerical.reshape(-1)
        analytical_values.extend(float(value) for value in analytical)
        numerical_values.extend(float(value) for value in numerical)
        relative_errors.extend(float(value) for value in result.relative_errors.reshape(-1))

        width = result.element_count
        tick_positions.append(offset + (width - 1) / 2.0)
        component_label = DISPLAY_COMPONENT_NAMES.get(
            result.component,
            result.component,
        )
        tick_labels.append(f"{component_label}\n{result.gradient_name}")
        offset += width
        boundaries.append(offset - 0.5)

    x_values = np.arange(len(analytical_values))
    error_floor = np.maximum(np.asarray(relative_errors), 1e-16)
    upper_error_limit = max(threshold * 10.0, float(error_floor.max()) * 10.0)

    figure, (value_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.08},
    )

    value_axis.plot(
        x_values,
        analytical_values,
        color="#1f77b4",
        linewidth=2.4,
        marker="o",
        markersize=4.0,
        markeredgewidth=0.0,
        label="Analytical",
        zorder=3,
    )
    value_axis.plot(
        x_values,
        numerical_values,
        color="#ff7f0e",
        linewidth=2.2,
        linestyle="--",
        marker="x",
        markersize=4.4,
        markeredgewidth=1.2,
        label="Numerical finite difference",
        zorder=4,
    )
    value_axis.set_ylabel("Gradient value")
    value_axis.set_title("Analytical vs Numerical Gradient Checks")
    value_axis.grid(True, alpha=0.32, linewidth=0.7, zorder=0)
    value_axis.legend(loc="best", frameon=True)

    error_axis.plot(
        x_values,
        error_floor,
        color="#2ca02c",
        linewidth=2.2,
        marker=".",
        markersize=5.0,
        label="Element relative error",
        zorder=3,
    )
    error_axis.axhline(
        threshold,
        color="#d62728",
        linewidth=2.3,
        linestyle="-",
        label=f"Threshold {threshold:g}",
        zorder=2,
    )
    error_axis.set_yscale("log")
    error_axis.set_ylim(1e-16, upper_error_limit)
    error_axis.set_ylabel("Relative error")
    error_axis.set_xlabel("Checked gradient entries")
    error_axis.grid(True, alpha=0.32, linewidth=0.7, which="both", zorder=0)
    error_axis.legend(loc="best", frameon=True)

    for boundary in boundaries[:-1]:
        value_axis.axvline(boundary, color="#4c4c4c", linewidth=1.0, alpha=0.35)
        error_axis.axvline(boundary, color="#4c4c4c", linewidth=1.0, alpha=0.35)

    error_axis.set_xticks(tick_positions)
    error_axis.set_xticklabels(tick_labels, fontsize=8)

    figure.savefig(figure_path, dpi=220)
    plt.close(figure)
    return figure_path


def generate_gradient_check_comparison(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    results = build_gradient_check_results()
    root = Path(output_dir)
    figure_path = plot_gradient_check_comparison(
        results,
        root / "gradient_check_comparison.png",
    )
    csv_path = write_gradient_check_csv(
        results,
        root / "gradient_check_comparison.csv",
    )
    summary_path = write_gradient_check_summary_json(
        results,
        root / "gradient_check_comparison_summary.json",
    )
    return {
        "figure": figure_path,
        "csv": csv_path,
        "summary": summary_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate analytical-vs-numerical gradient check artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated PNG, CSV, and JSON artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for name, path in generate_gradient_check_comparison(args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
