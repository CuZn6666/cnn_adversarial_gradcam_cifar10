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
    """Compact forward-only CNN for CIFAR-10 classification."""

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
        return self.classifier(features)

    __call__ = forward
