from __future__ import annotations

from typing import Any

import numpy as np

from configs.default_config import (
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    SEED,
)
from src.backend import ensure_backend_array, resolve_backend
from src.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLU


class CompactCNN:
    """Compact CNN with manual forward and backward passes."""

    def __init__(self, seed: int = SEED, backend: str | Any = "numpy") -> None:
        self.xp = resolve_backend(backend)
        rng = np.random.default_rng(seed)

        self.conv1 = Conv2D(
            IMAGE_CHANNELS,
            8,
            kernel_size=3,
            padding=1,
            rng=rng,
            backend=self.xp,
        )
        self.relu1 = ReLU(backend=self.xp)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2, backend=self.xp)

        self.conv2 = Conv2D(
            8,
            16,
            kernel_size=3,
            padding=1,
            rng=rng,
            backend=self.xp,
        )
        self.relu2 = ReLU(backend=self.xp)
        self.pool2 = MaxPool2D(kernel_size=2, stride=2, backend=self.xp)

        pooled_height = IMAGE_HEIGHT // 4
        pooled_width = IMAGE_WIDTH // 4
        self.flatten = Flatten(backend=self.xp)
        self.classifier = Linear(
            16 * pooled_height * pooled_width,
            NUM_CLASSES,
            rng=rng,
            backend=self.xp,
        )
        self._logits_shape: tuple[int, ...] | None = None
        self._gradcam_activation: Any | None = None
        self._backward_completed = False

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        ensure_backend_array(inputs, self.xp, name="inputs")
        expected_shape = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        if inputs.ndim != 4 or inputs.shape[1:] != expected_shape:
            raise ValueError(
                "CompactCNN expects input with shape "
                f"(N, {IMAGE_CHANNELS}, {IMAGE_HEIGHT}, {IMAGE_WIDTH})."
            )

        features = self.pool1(self.relu1(self.conv1(inputs)))
        features = self.conv2(features)
        features = self.relu2(features)
        self._gradcam_activation = features
        features = self.pool2(features)
        features = self.flatten(features)
        logits = self.classifier(features)
        self._logits_shape = logits.shape
        self._backward_completed = False
        return logits

    @property
    def gradcam_activation(self) -> np.ndarray:
        if self._gradcam_activation is None:
            raise RuntimeError(
                "CompactCNN.gradcam_activation requires a preceding forward call."
            )
        return self._gradcam_activation.copy()

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        ensure_backend_array(grad_logits, self.xp, name="grad_logits")
        if self._logits_shape is None:
            raise RuntimeError(
                "CompactCNN.backward requires a preceding forward call."
            )
        if grad_logits.shape != self._logits_shape:
            raise ValueError(
                "Logits gradient shape does not match CompactCNN output."
            )

        gradients = self.classifier.backward(grad_logits)
        gradients = self.flatten.backward(gradients)
        gradients = self.pool2.backward(gradients)
        gradients = self.relu2.backward(gradients)
        gradients = self.conv2.backward(gradients)
        gradients = self.pool1.backward(gradients)
        gradients = self.relu1.backward(gradients)
        grad_input = self.conv1.backward(gradients)
        self._backward_completed = True
        return grad_input

    def named_parameters_and_gradients(
        self,
    ) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
        if not self._backward_completed:
            raise RuntimeError(
                "CompactCNN.named_parameters_and_gradients requires "
                "a completed backward call."
            )

        return (
            ("conv1.weights", self.conv1.weights, self.conv1.grad_weight),
            ("conv1.bias", self.conv1.bias, self.conv1.grad_bias),
            ("conv2.weights", self.conv2.weights, self.conv2.grad_weight),
            ("conv2.bias", self.conv2.bias, self.conv2.grad_bias),
            (
                "classifier.weights",
                self.classifier.weights,
                self.classifier.grad_weight,
            ),
            (
                "classifier.bias",
                self.classifier.bias,
                self.classifier.grad_bias,
            ),
        )

    __call__ = forward
