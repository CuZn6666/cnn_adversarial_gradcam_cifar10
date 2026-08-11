import numpy as np

from src.backend import (
    backend_name,
    divide_where,
    get_array_module,
    resolve_backend,
    to_backend,
    to_numpy,
)
from src.checkpointing import load_checkpoint, save_checkpoint
from src.models import CompactCNN


def test_resolve_backend_defaults_to_numpy() -> None:
    assert resolve_backend("numpy") is np
    assert backend_name(resolve_backend("numpy")) == "numpy"


def test_to_backend_numpy_round_trip_preserves_values() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    backend_values = to_backend(values, "numpy")

    assert get_array_module(backend_values) is np
    np.testing.assert_array_equal(to_numpy(backend_values), values)


def test_divide_where_numpy_preserves_dtype_and_masked_output() -> None:
    numerator = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    denominator = np.array([2.0, 4.0, 0.0], dtype=np.float32)
    output = np.zeros_like(numerator)

    result = divide_where(
        numerator,
        denominator,
        out=output,
        where=denominator > 0,
    )

    assert result is output
    assert result.dtype == np.float32
    np.testing.assert_array_equal(
        result,
        np.array([0.5, 0.5, 0.0], dtype=np.float32),
    )


def test_compact_cnn_numpy_backend_matches_default_initialization() -> None:
    images = np.zeros((2, 3, 32, 32), dtype=np.float32)
    default_model = CompactCNN(seed=123)
    explicit_numpy_model = CompactCNN(seed=123, backend="numpy")

    default_logits = default_model.forward(images)
    explicit_logits = explicit_numpy_model.forward(images)

    np.testing.assert_allclose(explicit_logits, default_logits)


def test_checkpoint_round_trip_stays_numpy_compatible(tmp_path) -> None:
    source_model = CompactCNN(seed=42, backend="numpy")
    loaded_model = CompactCNN(seed=99, backend="numpy")
    checkpoint_path = tmp_path / "model.npz"

    save_checkpoint(source_model, checkpoint_path)
    load_checkpoint(loaded_model, checkpoint_path)

    parameter_pairs = (
        (source_model.conv1.weights, loaded_model.conv1.weights),
        (source_model.conv1.bias, loaded_model.conv1.bias),
        (source_model.conv2.weights, loaded_model.conv2.weights),
        (source_model.conv2.bias, loaded_model.conv2.bias),
        (source_model.classifier.weights, loaded_model.classifier.weights),
        (source_model.classifier.bias, loaded_model.classifier.bias),
    )
    for source, loaded in parameter_pairs:
        np.testing.assert_array_equal(source, loaded)
