import pytest
from matplotlib.axes import Axes

from src.plotting import plot_fgsm_accuracy_vs_epsilon, plot_metrics


def _metrics_history() -> list[dict[str, float | int]]:
    return [
        {
            "epoch": 1,
            "train_loss": 2.3,
            "train_accuracy": 0.12,
            "eval_loss": 2.2,
            "eval_accuracy": 0.15,
        },
        {
            "epoch": 2,
            "train_loss": 2.0,
            "train_accuracy": 0.25,
            "eval_loss": 2.1,
            "eval_accuracy": 0.20,
        },
    ]


def _fgsm_sweep_results() -> list[dict[str, float | int]]:
    return [
        {
            "epsilon": 0.1,
            "clean_accuracy": 0.75,
            "adversarial_accuracy": 0.25,
        },
        {
            "epsilon": 0.0,
            "clean_accuracy": 0.75,
            "adversarial_accuracy": 0.75,
        },
        {
            "epsilon": 0.05,
            "clean_accuracy": 0.75,
            "adversarial_accuracy": 0.5,
        },
    ]


def test_plot_metrics_creates_loss_and_accuracy_curve_files(tmp_path) -> None:
    output_dir = tmp_path / "nested" / "figures"

    loss_path, accuracy_path = plot_metrics(
        _metrics_history(),
        output_dir,
    )

    assert output_dir.is_dir()
    assert loss_path == output_dir / "loss_curve.png"
    assert accuracy_path == output_dir / "accuracy_curve.png"
    for figure_path in (loss_path, accuracy_path):
        assert figure_path.is_file()
        assert figure_path.stat().st_size > 0
        assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_metrics_supports_training_metric_aliases(tmp_path) -> None:
    metrics_history = [
        {"epoch": 1, "mean_loss": 1.5, "accuracy": 0.25},
        {"epoch": 2, "mean_loss": 1.0, "accuracy": 0.5},
    ]

    loss_path, accuracy_path = plot_metrics(
        metrics_history,
        tmp_path,
    )

    assert loss_path.is_file()
    assert accuracy_path.is_file()


def test_plot_metrics_rejects_empty_history(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="Metrics history must not be empty",
    ):
        plot_metrics([], tmp_path)


def test_plot_metrics_supports_filename_prefix(tmp_path) -> None:
    loss_path, accuracy_path = plot_metrics(
        _metrics_history(),
        tmp_path,
        filename_prefix="cifar10_subset_",
    )

    assert loss_path.name == "cifar10_subset_loss_curve.png"
    assert accuracy_path.name == "cifar10_subset_accuracy_curve.png"


def test_plot_fgsm_accuracy_vs_epsilon_creates_requested_file(
    tmp_path,
) -> None:
    output_path = tmp_path / "nested" / "figures" / "accuracy_vs_epsilon.png"

    saved_path = plot_fgsm_accuracy_vs_epsilon(
        _fgsm_sweep_results(),
        output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_fgsm_accuracy_vs_epsilon_preserves_input_order(
    tmp_path,
    monkeypatch,
) -> None:
    plotted_x_values = []
    original_plot = Axes.plot

    def record_plot(self, x_values, y_values, **kwargs):
        plotted_x_values.append(list(x_values))
        return original_plot(self, x_values, y_values, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)

    plot_fgsm_accuracy_vs_epsilon(
        _fgsm_sweep_results(),
        tmp_path / "accuracy_vs_epsilon.png",
    )

    assert plotted_x_values == [[0.1, 0.0, 0.05], [0.1, 0.0, 0.05]]


def test_plot_fgsm_accuracy_vs_epsilon_rejects_empty_results(
    tmp_path,
) -> None:
    with pytest.raises(
        ValueError,
        match="FGSM sweep results must not be empty",
    ):
        plot_fgsm_accuracy_vs_epsilon(
            [],
            tmp_path / "accuracy_vs_epsilon.png",
        )
