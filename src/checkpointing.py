from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.backend import get_array_module, to_backend, to_numpy
from src.models import CompactCNN

PARAMETER_NAMES = (
    "conv1.weights",
    "conv1.bias",
    "conv2.weights",
    "conv2.bias",
    "classifier.weights",
    "classifier.bias",
)


def _named_parameters(
    model: CompactCNN,
) -> tuple[tuple[str, Any], ...]:
    return (
        ("conv1.weights", model.conv1.weights),
        ("conv1.bias", model.conv1.bias),
        ("conv2.weights", model.conv2.weights),
        ("conv2.bias", model.conv2.bias),
        ("classifier.weights", model.classifier.weights),
        ("classifier.bias", model.classifier.bias),
    )


def save_checkpoint(model: CompactCNN, path: str | Path) -> Path:
    """Save CompactCNN trainable parameters to a NumPy archive."""
    checkpoint_path = Path(path)
    parameters = {
        name: to_numpy(parameter)
        for name, parameter in _named_parameters(model)
    }
    np.savez(checkpoint_path, **parameters)
    return checkpoint_path


def load_checkpoint(model: CompactCNN, path: str | Path) -> None:
    """Load CompactCNN trainable parameters from a NumPy archive."""
    checkpoint_path = Path(path)

    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        missing_keys = [
            name for name in PARAMETER_NAMES if name not in checkpoint
        ]
        if missing_keys:
            raise ValueError(
                "Checkpoint is missing parameter keys: "
                + ", ".join(missing_keys)
            )

        target_parameters = dict(_named_parameters(model))
        loaded_parameters = {
            name: checkpoint[name].copy()
            for name in PARAMETER_NAMES
        }

    for name in PARAMETER_NAMES:
        expected_shape = target_parameters[name].shape
        loaded_shape = loaded_parameters[name].shape
        if loaded_shape != expected_shape:
            raise ValueError(
                f"Checkpoint parameter shape mismatch for {name}: "
                f"expected {expected_shape}, got {loaded_shape}."
            )

    for name in PARAMETER_NAMES:
        backend = get_array_module(target_parameters[name])
        target_parameters[name][...] = to_backend(
            loaded_parameters[name],
            backend,
        )
