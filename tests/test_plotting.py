import pytest
from matplotlib.axes import Axes

from src.plotting import (
    plot_confusion_matrix,
    plot_fgsm_accuracy_drop_vs_epsilon,
    plot_fgsm_attack_success_rate_vs_epsilon,
    plot_fgsm_accuracy_vs_epsilon,
    plot_fgsm_portfolio_accuracy_vs_epsilon,
    plot_metrics,
    plot_runtime_comparison,
    plot_train_validation_accuracy_curve,
    plot_training_loss_curve,
    runtime_speedup,
)


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


def _portfolio_history() -> list[dict[str, float | int]]:
    return [
        {
            "epoch": 1,
            "training_loss": 2.2,
            "training_accuracy": 0.18,
            "validation_accuracy": 0.16,
        },
        {
            "epoch": 2,
            "training_loss": 1.9,
            "training_accuracy": 0.31,
            "validation_accuracy": 0.27,
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


def _portfolio_fgsm_sweep_results() -> list[dict[str, float | int]]:
    return [
        {
            "epsilon": 0.0,
            "clean_accuracy": 0.5,
            "adversarial_accuracy": 0.5,
            "accuracy_drop": 0.0,
            "attack_success_rate": 0.0,
        },
        {
            "epsilon": 8.0 / 255.0,
            "clean_accuracy": 0.5,
            "adversarial_accuracy": 0.25,
            "accuracy_drop": 0.25,
            "attack_success_rate": 0.5,
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


def test_runtime_speedup_calculates_before_after_ratio() -> None:
    assert runtime_speedup(0.043458736, 0.000209222) == pytest.approx(
        207.7159,
    )


def test_plot_runtime_comparison_creates_requested_file(tmp_path) -> None:
    output_path = tmp_path / "runtime" / "conv2d_backward.png"

    saved_path = plot_runtime_comparison(
        0.043458736,
        0.000209222,
        output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("before_seconds", "after_seconds", "message"),
    [
        (0.0, 0.1, "before_seconds must be"),
        (-1.0, 0.1, "before_seconds must be"),
        (0.1, 0.0, "after_seconds must be"),
        (0.1, -1.0, "after_seconds must be"),
    ],
)
def test_plot_runtime_comparison_rejects_invalid_timings(
    tmp_path,
    before_seconds,
    after_seconds,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        plot_runtime_comparison(
            before_seconds,
            after_seconds,
            tmp_path / "runtime.png",
        )


def test_plot_training_loss_curve_creates_requested_file(tmp_path) -> None:
    output_path = tmp_path / "baseline" / "training_loss_curve.png"

    saved_path = plot_training_loss_curve(_portfolio_history(), output_path)

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_train_validation_accuracy_curve_creates_requested_file(
    tmp_path,
) -> None:
    output_path = tmp_path / "baseline" / "accuracy_curve.png"

    saved_path = plot_train_validation_accuracy_curve(
        _portfolio_history(),
        output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_confusion_matrix_creates_requested_file(tmp_path) -> None:
    output_path = tmp_path / "baseline" / "confusion_matrix.png"

    saved_path = plot_confusion_matrix(
        confusion_matrix=[[3, 1], [0, 4]],
        class_names=["class_a", "class_b"],
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_confusion_matrix_rejects_invalid_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-negative finite"):
        plot_confusion_matrix(
            confusion_matrix=[[1, -1], [0, 2]],
            class_names=["class_a", "class_b"],
            output_path=tmp_path / "confusion_matrix.png",
        )


@pytest.mark.parametrize(
    ("plot_function", "filename"),
    [
        (plot_fgsm_portfolio_accuracy_vs_epsilon, "accuracy_vs_epsilon.png"),
        (
            plot_fgsm_attack_success_rate_vs_epsilon,
            "attack_success_rate_vs_epsilon.png",
        ),
        (
            plot_fgsm_accuracy_drop_vs_epsilon,
            "accuracy_drop_vs_epsilon.png",
        ),
    ],
)
def test_portfolio_fgsm_plots_create_requested_files(
    tmp_path,
    plot_function,
    filename,
) -> None:
    output_path = tmp_path / "fgsm" / filename

    saved_path = plot_function(
        _portfolio_fgsm_sweep_results(),
        output_path,
        epsilon_labels=("0", "8/255"),
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_portfolio_fgsm_plots_reject_empty_results(tmp_path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        plot_fgsm_attack_success_rate_vs_epsilon(
            [],
            tmp_path / "attack_success_rate_vs_epsilon.png",
        )


def test_portfolio_fgsm_plots_reject_label_length_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError, match="epsilon_labels length"):
        plot_fgsm_accuracy_drop_vs_epsilon(
            _portfolio_fgsm_sweep_results(),
            tmp_path / "accuracy_drop_vs_epsilon.png",
            epsilon_labels=("0",),
        )
