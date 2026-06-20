from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Conv2D:
    """2D convolution with manual forward and backward passes for NCHW inputs."""

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
        self.grad_weight = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self.padding = padding
        self.stride = stride
        self._input_shape: tuple[int, ...] | None = None
        self._output_shape: tuple[int, ...] | None = None
        self._padded_inputs: np.ndarray | None = None

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
        outputs = outputs.astype(np.float32, copy=False)

        self._input_shape = inputs.shape
        self._output_shape = outputs.shape
        self._padded_inputs = padded

        return outputs

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if (
            self._input_shape is None
            or self._output_shape is None
            or self._padded_inputs is None
        ):
            raise RuntimeError("Conv2D.backward requires a preceding forward call.")
        if grad_out.shape != self._output_shape:
            raise ValueError("Output gradient shape does not match Conv2D output.")

        kernel_height, kernel_width = self.weights.shape[2:]
        grad_padded_input = np.zeros_like(
            self._padded_inputs,
            dtype=grad_out.dtype,
        )
        self.grad_weight = np.zeros_like(self.weights, dtype=grad_out.dtype)
        self.grad_bias = grad_out.sum(axis=(0, 2, 3))

        for batch_index in range(self._input_shape[0]):
            for output_channel in range(self.weights.shape[0]):
                for output_row in range(self._output_shape[2]):
                    input_row = output_row * self.stride
                    for output_column in range(self._output_shape[3]):
                        input_column = output_column * self.stride
                        gradient = grad_out[
                            batch_index,
                            output_channel,
                            output_row,
                            output_column,
                        ]
                        input_window = self._padded_inputs[
                            batch_index,
                            :,
                            input_row : input_row + kernel_height,
                            input_column : input_column + kernel_width,
                        ]

                        self.grad_weight[output_channel] += (
                            gradient * input_window
                        )
                        grad_padded_input[
                            batch_index,
                            :,
                            input_row : input_row + kernel_height,
                            input_column : input_column + kernel_width,
                        ] += gradient * self.weights[output_channel]

        if self.padding == 0:
            return grad_padded_input

        return grad_padded_input[
            :,
            :,
            self.padding : -self.padding,
            self.padding : -self.padding,
        ]

    __call__ = forward


class ReLU:
    """Rectified linear activation with manual forward and backward passes."""

    def __init__(self) -> None:
        self._positive_mask: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._positive_mask = inputs > 0
        return np.maximum(inputs, 0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._positive_mask is None:
            raise RuntimeError("ReLU.backward requires a preceding forward call.")
        if grad_out.shape != self._positive_mask.shape:
            raise ValueError("Output gradient shape does not match ReLU output.")

        return grad_out * self._positive_mask

    __call__ = forward


class MaxPool2D:
    """Square max pooling with manual forward and backward passes."""

    def __init__(self, kernel_size: int = 2, stride: int | None = None) -> None:
        if kernel_size <= 0:
            raise ValueError("Pooling kernel size must be positive.")

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self._input_shape: tuple[int, ...] | None = None
        self._output_shape: tuple[int, ...] | None = None
        self._max_indices: np.ndarray | None = None

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
        outputs = windows.max(axis=(-2, -1))

        self._input_shape = inputs.shape
        self._output_shape = outputs.shape
        flattened_windows = windows.reshape(*windows.shape[:4], -1)
        # np.argmax selects the first maximum in row-major order for ties.
        self._max_indices = flattened_windows.argmax(axis=-1)

        return outputs

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if (
            self._input_shape is None
            or self._output_shape is None
            or self._max_indices is None
        ):
            raise RuntimeError("MaxPool2D.backward requires a preceding forward call.")
        if grad_out.shape != self._output_shape:
            raise ValueError("Output gradient shape does not match MaxPool2D output.")

        grad_input = np.zeros(self._input_shape, dtype=grad_out.dtype)
        batch_indices, channel_indices = np.indices(self._input_shape[:2])

        for output_row in range(self._output_shape[2]):
            for output_column in range(self._output_shape[3]):
                max_indices = self._max_indices[
                    :, :, output_row, output_column
                ]
                input_rows = (
                    output_row * self.stride
                    + max_indices // self.kernel_size
                )
                input_columns = (
                    output_column * self.stride
                    + max_indices % self.kernel_size
                )
                np.add.at(
                    grad_input,
                    (
                        batch_indices,
                        channel_indices,
                        input_rows,
                        input_columns,
                    ),
                    grad_out[:, :, output_row, output_column],
                )

        return grad_input

    __call__ = forward


class Flatten:
    """Flatten layer with manual forward and backward passes."""

    def __init__(self) -> None:
        self._input_shape: tuple[int, ...] | None = None
        self._output_shape: tuple[int, ...] | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim < 2:
            raise ValueError("Flatten expects an input with a batch dimension.")

        self._input_shape = inputs.shape
        outputs = inputs.reshape(inputs.shape[0], -1)
        self._output_shape = outputs.shape
        return outputs

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._input_shape is None or self._output_shape is None:
            raise RuntimeError("Flatten.backward requires a preceding forward call.")
        if grad_out.shape != self._output_shape:
            raise ValueError("Output gradient shape does not match Flatten output.")

        return grad_out.reshape(self._input_shape)

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
