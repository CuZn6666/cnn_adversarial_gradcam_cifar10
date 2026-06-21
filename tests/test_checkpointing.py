import numpy as np
import pytest

from src.checkpointing import (
    PARAMETER_NAMES,
    load_checkpoint,
    save_checkpoint,
)
from src.models import CompactCNN


def _model_parameters(
    model: CompactCNN,
) -> tuple[tuple[str, np.ndarray], ...]:
    return (
        ("conv1.weights", model.conv1.weights),
        ("conv1.bias", model.conv1.bias),
        ("conv2.weights", model.conv2.weights),
        ("conv2.bias", model.conv2.bias),
        ("classifier.weights", model.classifier.weights),
        ("classifier.bias", model.classifier.bias),
    )


def test_checkpoint_round_trip_restores_parameters_and_logits(tmp_path) -> None:
    checkpoint_path = tmp_path / "compact_cnn.npz"
    inputs = np.random.default_rng(7).random(
        (2, 3, 32, 32),
        dtype=np.float32,
    )
    source_model = CompactCNN(seed=42)
    expected_logits = source_model.forward(inputs)

    saved_path = save_checkpoint(source_model, checkpoint_path)
    loaded_model = CompactCNN(seed=99)
    load_checkpoint(loaded_model, checkpoint_path)
    loaded_logits = loaded_model.forward(inputs)

    assert saved_path == checkpoint_path
    assert checkpoint_path.is_file()
    for (source_name, source_parameter), (
        loaded_name,
        loaded_parameter,
    ) in zip(
        _model_parameters(source_model),
        _model_parameters(loaded_model),
    ):
        assert source_name == loaded_name
        np.testing.assert_array_equal(loaded_parameter, source_parameter)
    np.testing.assert_array_equal(loaded_logits, expected_logits)


def test_checkpoint_save_and_load_do_not_require_gradients(tmp_path) -> None:
    checkpoint_path = tmp_path / "without_gradients.npz"
    source_model = CompactCNN(seed=42)
    loaded_model = CompactCNN(seed=99)

    save_checkpoint(source_model, checkpoint_path)
    load_checkpoint(loaded_model, checkpoint_path)

    for (_, source_parameter), (_, loaded_parameter) in zip(
        _model_parameters(source_model),
        _model_parameters(loaded_model),
    ):
        np.testing.assert_array_equal(loaded_parameter, source_parameter)


def test_checkpoint_load_rejects_missing_parameter_keys(tmp_path) -> None:
    checkpoint_path = tmp_path / "missing_key.npz"
    model = CompactCNN(seed=42)
    parameters = dict(_model_parameters(model))
    parameters.pop("classifier.bias")
    np.savez(checkpoint_path, **parameters)

    with pytest.raises(
        ValueError,
        match="Checkpoint is missing parameter keys: classifier.bias",
    ):
        load_checkpoint(CompactCNN(seed=99), checkpoint_path)


def test_checkpoint_load_rejects_shape_mismatch(tmp_path) -> None:
    checkpoint_path = tmp_path / "wrong_shape.npz"
    model = CompactCNN(seed=42)
    parameters = dict(_model_parameters(model))
    parameters["conv1.weights"] = np.zeros(
        (1, 1, 1, 1),
        dtype=np.float32,
    )
    np.savez(checkpoint_path, **parameters)

    with pytest.raises(
        ValueError,
        match="Checkpoint parameter shape mismatch for conv1.weights",
    ):
        load_checkpoint(CompactCNN(seed=99), checkpoint_path)


def test_checkpoint_contains_exactly_six_trainable_parameters(tmp_path) -> None:
    checkpoint_path = tmp_path / "keys.npz"

    save_checkpoint(CompactCNN(seed=42), checkpoint_path)

    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        assert tuple(checkpoint.files) == PARAMETER_NAMES
