from pathlib import Path

import numpy as np
import pytest

from experiments.fgsm import evaluate_robustness
from src.metrics import load_metrics


def _tiny_config(tmp_path: Path) -> evaluate_robustness.WP8FGSMRobustnessConfig:
    return evaluate_robustness.WP8FGSMRobustnessConfig(
        eval_samples=3,
        batch_size=2,
        seed=17,
        epsilon_values=(0.0, 0.05),
        output_dir=tmp_path / "results",
        representative_epsilon=0.05,
        max_successful_examples=1,
        max_failed_examples=1,
        checkpoint_path=tmp_path / "unused_checkpoint.npz",
    )


def _sweep_results() -> list[dict[str, float | int]]:
    return [
        {
            "epsilon": 0.0,
            "total_samples": 3,
            "clean_correct": 2,
            "adversarial_correct": 2,
            "clean_correct_samples": 2,
            "successful_attacks": 0,
            "clean_accuracy": 2.0 / 3.0,
            "adversarial_accuracy": 2.0 / 3.0,
            "accuracy_drop": 0.0,
            "attack_success_rate": 0.0,
        },
        {
            "epsilon": 0.05,
            "total_samples": 3,
            "clean_correct": 2,
            "adversarial_correct": 1,
            "clean_correct_samples": 2,
            "successful_attacks": 1,
            "clean_accuracy": 2.0 / 3.0,
            "adversarial_accuracy": 1.0 / 3.0,
            "accuracy_drop": 1.0 / 3.0,
            "attack_success_rate": 0.5,
        },
    ]


def _representative_examples() -> dict[str, list[dict[str, object]]]:
    return {
        "successful": [
            {
                "global_sample_index": 0,
                "batch_index": 0,
                "index_in_batch": 0,
                "true_label": 0,
                "clean_prediction": 0,
                "adversarial_prediction": 1,
                "epsilon": 0.05,
                "example_type": "successful",
            }
        ],
        "failed": [],
    }


def test_wp8_config_defaults_match_documented_controlled_settings() -> None:
    config = evaluate_robustness.WP8FGSMRobustnessConfig()

    assert config.eval_samples == 32
    assert config.batch_size == 8
    assert config.seed == 42
    assert config.epsilon_values == tuple(index / 255.0 for index in range(17))
    assert config.output_dir.name == "WP8"
    assert config.representative_epsilon == 8.0 / 255.0
    assert config.max_successful_examples == 1
    assert config.max_failed_examples == 1


def test_run_fgsm_robustness_pipeline_creates_metrics_and_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(tmp_path)
    batches = (
        (
            np.zeros((1, 3, 32, 32), dtype=np.float32),
            np.array([0], dtype=np.int64),
        ),
        (
            np.zeros((2, 3, 32, 32), dtype=np.float32),
            np.array([1, 2], dtype=np.int64),
        ),
    )
    expected_sweep = _sweep_results()
    expected_examples = _representative_examples()
    calls: dict[str, object] = {}

    def fake_sweep(model, loss_function, received_batches, epsilons):
        calls["sweep_batches"] = received_batches
        calls["epsilons"] = epsilons
        return expected_sweep

    def fake_selection(
        model,
        loss_function,
        received_batches,
        epsilon,
        max_successful,
        max_failed,
    ):
        calls["selection_batches"] = received_batches
        calls["representative_epsilon"] = epsilon
        calls["limits"] = (max_successful, max_failed)
        return expected_examples

    def fake_plot(sweep_results, output_path):
        calls["plot_results"] = sweep_results
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return path

    monkeypatch.setattr(
        evaluate_robustness,
        "evaluate_fgsm_epsilon_sweep",
        fake_sweep,
    )
    monkeypatch.setattr(
        evaluate_robustness,
        "select_fgsm_representative_examples",
        fake_selection,
    )
    monkeypatch.setattr(
        evaluate_robustness,
        "plot_fgsm_accuracy_vs_epsilon",
        fake_plot,
    )

    result = evaluate_robustness.run_fgsm_robustness_pipeline(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        iter(batches),
        config,
    )

    assert result["metrics_path"].is_file()
    assert result["plot_path"].is_file()
    assert result["sweep_results"] == expected_sweep
    assert result["representative_examples"] == expected_examples
    saved_metrics = load_metrics(result["metrics_path"])
    assert saved_metrics["sweep_results"] == expected_sweep
    assert saved_metrics["representative_examples"] == expected_examples

    assert calls["sweep_batches"] is calls["selection_batches"]
    assert calls["epsilons"] == config.epsilon_values
    assert calls["representative_epsilon"] == config.representative_epsilon
    assert calls["limits"] == (1, 1)
    assert calls["plot_results"] == expected_sweep


def test_cifar10_runner_does_not_load_data_without_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(tmp_path)

    def fail_if_loaded(data_dir):
        raise AssertionError(f"Unexpected CIFAR-10 loader call for {data_dir}.")

    monkeypatch.setattr(evaluate_robustness, "load_cifar10", fail_if_loaded)

    with pytest.raises(FileNotFoundError, match="checkpoint is not available"):
        evaluate_robustness.run_cifar10_fgsm_robustness(
            config,
            data_dir=tmp_path / "missing_data",
        )
