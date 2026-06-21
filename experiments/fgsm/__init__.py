"""Controlled WP7 FGSM example generation."""

from typing import Any

__all__ = [
    "generate_fgsm_examples",
    "run_cifar10_fgsm_examples",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from experiments.fgsm import generate_examples

    return getattr(generate_examples, name)
