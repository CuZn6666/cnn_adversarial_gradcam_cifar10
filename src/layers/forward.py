from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Conv2D:
    """Forward-only 2D convolution for NCHW inputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        if min(in_channels, out_channels, kernel_size, stride) <= 0:
            raise ValueError("Convolution dimensions and stride must be positive.")
        if padding < 0:
            raise ValueError("Padding must be non-negative.")

        generator = rng if rng is not None else np.random.default_rng()
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        self.weights = generator.normal(
            0.0,
            scale,
            size=(out_channels, in_channels, kernel_size, kernel_size),
        ).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)
        self.padding = padding
        self.stride = stride

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 4:
            raise ValueError("Conv2D expects input with shape (N, C, H, W).")
        if inputs.shape[1] != self.weights.shape[1]:
            raise ValueError("Input channels do not match convolution weights.")

        kernel_height, kernel_width = self.weights.shape[2:]
        padded = np.pad(
            inputs,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
        )

        if padded.shape[2] < kernel_height or padded.shape[3] < kernel_width:
            raise ValueError("Kernel size is larger than the padded input.")

        windows = sliding_window_view(
            padded,
            (kernel_height, kernel_width),
            axis=(2, 3),
        )
        windows = windows[:, :, :: self.stride, :: self.stride, :, :]

        outputs = np.einsum(
            "nchwkl,ockl->nohw",
            windows,
            self.weights,
            optimize=True,
        )
        outputs += self.bias[None, :, None, None]
        return outputs.astype(np.float32, copy=False)

    __call__ = forward


class ReLU:
    """Forward-only rectified linear activation."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.maximum(inputs, 0)

    __call__ = forward


class MaxPool2D:
    """Forward-only square max pooling for NCHW inputs."""

    def __init__(self, kernel_size: int = 2, stride: int | None = None) -> None:
        if kernel_size <= 0:
            raise ValueError("Pooling kernel size must be positive.")

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

        if self.stride <= 0:
            raise ValueError("Pooling stride must be positive.")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 4:
            raise ValueError("MaxPool2D expects input with shape (N, C, H, W).")
        if (
            inputs.shape[2] < self.kernel_size
            or inputs.shape[3] < self.kernel_size
        ):
            raise ValueError("Pooling kernel size is larger than the input.")

        windows = sliding_window_view(
            inputs,
            (self.kernel_size, self.kernel_size),
            axis=(2, 3),
        )
        windows = windows[:, :, :: self.stride, :: self.stride, :, :]
        return windows.max(axis=(-2, -1))

    __call__ = forward


class Flatten:
    """Flatten all dimensions except the batch dimension."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim < 2:
            raise ValueError("Flatten expects an input with a batch dimension.")
        return inputs.reshape(inputs.shape[0], -1)

    __call__ = forward


class Linear:
    """Fully connected layer with manual forward and backward passes."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if min(in_features, out_features) <= 0:
            raise ValueError("Linear layer dimensions must be positive.")

        generator = rng if rng is not None else np.random.default_rng()
        scale = np.sqrt(2.0 / in_features)

        self.weights = generator.normal(
            0.0,
            scale,
            size=(out_features, in_features),
        ).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.grad_weight = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self._inputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 2:
            raise ValueError("Linear expects input with shape (N, features).")
        if inputs.shape[1] != self.weights.shape[1]:
            raise ValueError("Input features do not match linear weights.")

        self._inputs = inputs
        return inputs @ self.weights.T + self.bias

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._inputs is None:
            raise RuntimeError("Linear.backward requires a preceding forward call.")
        if grad_out.ndim != 2:
            raise ValueError("Linear backward expects shape (N, out_features).")
        if grad_out.shape != (self._inputs.shape[0], self.weights.shape[0]):
            raise ValueError("Output gradient shape does not match Linear output.")

        self.grad_weight = grad_out.T @ self._inputs
        self.grad_bias = grad_out.sum(axis=0)
        return grad_out @ self.weights

    __call__ = forward
