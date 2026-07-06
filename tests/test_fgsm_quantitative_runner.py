from pathlib import Path

import numpy as np
import pytest

from experiments.fgsm import evaluate_quantitative
from src.metrics import load_metrics


def _tiny_config(tmp_path: Path) -> (
    evaluate_quantitative.PortfolioFGSMQuantitativeConfig
):
    return evaluate_quantitative.PortfolioFGSMQuantitativeConfig(
        eval_samples=4,
        batch_size=2,
        seed=17,
        epsilon_values=(0.0, 0.05),
        epsilon_labels=("0", "0.05"),
        checkpoint_path=tmp_path / "unused_checkpoint.npz",
        output_dir=tmp_path / "fgsm",
    )


def _sweep_results() -> list[dict[str, float | int]]:
    return [
        {
            "epsilon": 0.0,
            "total_samples": 4,
            "clean_correct": 2,
            "adversarial_correct": 2,
            "clean_correct_samples": 2,
            "successful_attacks": 0,
            "clean_accuracy": 0.5,
            "adversarial_accuracy": 0.5,
            "accuracy_drop": 0.0,
            "attack_success_rate": 0.0,
        },
        {
            "epsilon": 0.05,
            "total_samples": 4,
            "clean_correct": 2,
            "adversarial_correct": 1,
            "clean_correct_samples": 2,
            "successful_attacks": 1,
            "clean_accuracy": 0.5,
            "adversarial_accuracy": 0.25,
            "accuracy_drop": 0.25,
            "attack_success_rate": 0.5,
        },
    ]


def test_portfolio_fgsm_config_defaults_match_day2_requirements() -> None:
    config = evaluate_quantitative.PortfolioFGSMQuantitativeConfig()

    assert config.eval_samples == 1024
    assert config.batch_size == 32
    assert config.seed == 42
    assert config.epsilon_values == (
        0.0,
        2.0 / 255.0,
        4.0 / 255.0,
        8.0 / 255.0,
        16.0 / 255.0,
    )
    assert config.epsilon_labels == (
        "0",
        "2/255",
        "4/255",
        "8/255",
        "16/255",
    )
    assert config.checkpoint_path.name == "portfolio_baseline_best.npz"
    assert config.output_dir.name == "fgsm"


def test_portfolio_fgsm_config_rejects_label_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_quantitative.PortfolioFGSMQuantitativeConfig(
            eval_samples=4,
            batch_size=2,
            epsilon_values=(0.0, 0.05),
            epsilon_labels=("0",),
            checkpoint_path=tmp_path / "checkpoint.npz",
            output_dir=tmp_path,
        )


def test_run_portfolio_fgsm_quantitative_pipeline_creates_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(tmp_path)
    batches = (
        (
            np.zeros((2, 3, 32, 32), dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
        ),
    )
    expected_sweep = _sweep_results()
    calls: dict[str, object] = {}

    def fake_sweep(model, loss_function, received_batches, epsilons):
        calls["batches"] = received_batches
        calls["epsilons"] = epsilons
        return expected_sweep

    monkeypatch.setattr(
        evaluate_quantitative,
        "evaluate_fgsm_epsilon_sweep",
        fake_sweep,
    )

    result = evaluate_quantitative.run_portfolio_fgsm_quantitative_pipeline(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        iter(batches),
        config,
    )

    assert result["metrics_path"].is_file()
    assert result["accuracy_plot_path"].is_file()
    assert result["attack_success_rate_plot_path"].is_file()
    assert result["accuracy_drop_plot_path"].is_file()
    assert result["sweep_results"] == expected_sweep
    saved_metrics = load_metrics(result["metrics_path"])
    assert saved_metrics["config"]["epsilon_labels"] == ["0", "0.05"]
    assert saved_metrics["sweep_results"] == expected_sweep
    assert calls["epsilons"] == config.epsilon_values


def test_portfolio_fgsm_pipeline_rejects_epsilon_zero_metric_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(tmp_path)
    invalid_sweep = _sweep_results()
    invalid_sweep[0] = {
        **invalid_sweep[0],
        "adversarial_correct": 1,
        "adversarial_accuracy": 0.25,
    }

    monkeypatch.setattr(
        evaluate_quantitative,
        "evaluate_fgsm_epsilon_sweep",
        lambda model, loss_function, batches, epsilons: invalid_sweep,
    )

    with pytest.raises(RuntimeError, match="epsilon=0"):
        evaluate_quantitative.run_portfolio_fgsm_quantitative_pipeline(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            (),
            config,
        )


def test_portfolio_cifar10_runner_does_not_load_data_without_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(tmp_path)

    def fail_if_loaded(data_dir):
        raise AssertionError(f"Unexpected CIFAR-10 loader call for {data_dir}.")

    monkeypatch.setattr(evaluate_quantitative, "load_cifar10", fail_if_loaded)

    with pytest.raises(FileNotFoundError, match="checkpoint is not available"):
        evaluate_quantitative.run_portfolio_cifar10_fgsm_quantitative(
            config,
            data_dir=tmp_path / "missing_data",
        )
