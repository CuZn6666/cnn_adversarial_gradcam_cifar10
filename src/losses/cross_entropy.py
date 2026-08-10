from __future__ import annotations

from typing import Any

import numpy as np

from src.backend import (
    ensure_backend_array,
    resolve_backend,
    to_python_bool,
    to_python_float,
)


class SoftmaxCrossEntropyLoss:
    """Mean softmax cross-entropy loss for class-index labels."""

    def __init__(self, backend: str | Any = "numpy") -> None:
        self.xp = resolve_backend(backend)
        self._probabilities: Any | None = None
        self._labels: Any | None = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        xp = ensure_backend_array(logits, self.xp, name="logits")
        ensure_backend_array(labels, self.xp, name="labels")
        if logits.ndim != 2:
            raise ValueError(
                "SoftmaxCrossEntropyLoss expects logits with shape "
                "(batch_size, num_classes)."
            )
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError(
                "SoftmaxCrossEntropyLoss expects labels with shape "
                "(batch_size,)."
            )
        if logits.shape[0] == 0 or logits.shape[1] == 0:
            raise ValueError("SoftmaxCrossEntropyLoss requires non-empty logits.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("SoftmaxCrossEntropyLoss labels must be integers.")
        if to_python_bool(xp.any(labels < 0)) or to_python_bool(
            xp.any(labels >= logits.shape[1])
        ):
            raise ValueError(
                "SoftmaxCrossEntropyLoss labels are outside the valid class range."
            )

        shifted_logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = xp.exp(shifted_logits)
        exp_sums = exp_logits.sum(axis=1, keepdims=True)
        probabilities = exp_logits / exp_sums
        log_probabilities = shifted_logits - xp.log(exp_sums)
        batch_indices = xp.arange(logits.shape[0])
        loss = -log_probabilities[batch_indices, labels].mean()

        self._probabilities = probabilities
        self._labels = labels.copy()

        return to_python_float(loss)

    def backward(self) -> np.ndarray:
        if self._probabilities is None or self._labels is None:
            raise RuntimeError(
                "SoftmaxCrossEntropyLoss.backward requires a preceding "
                "forward call."
            )

        grad_logits = self._probabilities.copy()
        batch_indices = self.xp.arange(grad_logits.shape[0])
        grad_logits[batch_indices, self._labels] -= 1.0
        grad_logits /= grad_logits.shape[0]
        return grad_logits
