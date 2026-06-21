from __future__ import annotations

import numpy as np

from configs.default_config import (
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    SEED,
)
from src.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLU


class CompactCNN:
    """Compact CNN with manual forward and backward passes."""

    def __init__(self, seed: int = SEED) -> None:
        rng = np.random.default_rng(seed)

        self.conv1 = Conv2D(IMAGE_CHANNELS, 8, kernel_size=3, padding=1, rng=rng)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = Conv2D(8, 16, kernel_size=3, padding=1, rng=rng)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)

        pooled_height = IMAGE_HEIGHT // 4
        pooled_width = IMAGE_WIDTH // 4
        self.flatten = Flatten()
        self.classifier = Linear(
            16 * pooled_height * pooled_width,
            NUM_CLASSES,
            rng=rng,
        )
        self._logits_shape: tuple[int, ...] | None = None
        self._backward_completed = False

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        expected_shape = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        if inputs.ndim != 4 or inputs.shape[1:] != expected_shape:
            raise ValueError(
                "CompactCNN expects input with shape "
                f"(N, {IMAGE_CHANNELS}, {IMAGE_HEIGHT}, {IMAGE_WIDTH})."
            )

        features = self.pool1(self.relu1(self.conv1(inputs)))
        features = self.pool2(self.relu2(self.conv2(features)))
        features = self.flatten(features)
        logits = self.classifier(features)
        self._logits_shape = logits.shape
        self._backward_completed = False
        return logits

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
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
