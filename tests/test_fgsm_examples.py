from pathlib import Path

import numpy as np

from configs.default_config import PROJECT_ROOT
from experiments.fgsm.generate_examples import (
    DEFAULT_OUTPUT_DIR,
    generate_fgsm_examples,
)
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN


def _model_parameters(model: CompactCNN) -> tuple[np.ndarray, ...]:
    return (
        model.conv1.weights,
        model.conv1.bias,
        model.conv2.weights,
        model.conv2.bias,
        model.classifier.weights,
        model.classifier.bias,
    )


def test_generate_fgsm_examples_creates_deterministic_artifacts(
    tmp_path: Path,
) -> None:
    model = CompactCNN(seed=42)
    loss_function = SoftmaxCrossEntropyLoss()
    images = np.random.default_rng(17).random(
        (2, 3, 32, 32),
        dtype=np.float32,
    )
    labels = np.array([2, 5], dtype=np.int64)
    parameters_before = [
        parameter.copy() for parameter in _model_parameters(model)
    ]

    first_results = generate_fgsm_examples(
        model,
        loss_function,
        images,
        labels,
        tmp_path / "figures",
        epsilon=0.05,
        example_count=1,
    )
    second_results = generate_fgsm_examples(
        model,
        loss_function,
        images,
        labels,
        tmp_path / "figures",
        epsilon=0.05,
        example_count=1,
    )

    assert len(first_results) == 1
    assert first_results == second_results
    assert first_results[0]["example_index"] == 0
    assert first_results[0]["label"] == 2
    assert first_results[0]["epsilon"] == 0.05

    paths = first_results[0]["paths"]
    assert {path.name for path in paths.values()} == {
        "fgsm_example_000_clean.png",
        "fgsm_example_000_adversarial.png",
        "fgsm_example_000_input_gradient.png",
        "fgsm_example_000_perturbation.png",
    }
    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0

    for parameter, parameter_before in zip(
        _model_parameters(model),
        parameters_before,
        strict=True,
    ):
        np.testing.assert_array_equal(parameter, parameter_before)


def test_default_output_directory_is_wp7_qualitative_results() -> None:
    assert DEFAULT_OUTPUT_DIR == (
        PROJECT_ROOT / "results" / "WP7" / "qualitative"
    )
