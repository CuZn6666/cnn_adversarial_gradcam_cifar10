from __future__ import annotations

from pathlib import Path

from src.plotting import plot_runtime_comparison
from src.visualization import save_combined_fgsm_figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Historical WP6 benchmark values from:
# deliverables/WP6/runtime_benchmark_after.md
CONV2D_BACKWARD_BEFORE_SECONDS = 0.043458736
CONV2D_BACKWARD_AFTER_SECONDS = 0.000209222


def generate_portfolio_figures(
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Path]:
    """Generate README-ready portfolio figures from committed artifacts."""
    root = Path(project_root)

    combined_fgsm_path = save_combined_fgsm_figure(
        root / "results/WP7/qualitative/fgsm_example_000_clean.png",
        root / "results/WP7/qualitative/fgsm_example_000_adversarial.png",
        root / "results/WP7/qualitative/fgsm_example_000_input_gradient.png",
        root / "results/WP7/qualitative/fgsm_example_000_perturbation.png",
        root / "results/WP7/qualitative/fgsm_example_000_combined.png",
    )

    runtime_path = plot_runtime_comparison(
        CONV2D_BACKWARD_BEFORE_SECONDS,
        CONV2D_BACKWARD_AFTER_SECONDS,
        root / "results/WP6/conv2d_backward_runtime_comparison.png",
        operation="Conv2D.backward",
    )

    return {
        "combined_fgsm": combined_fgsm_path,
        "runtime_comparison": runtime_path,
    }


def main() -> None:
    for name, path in generate_portfolio_figures().items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
