import pytest

from src.plotting import plot_metrics


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
