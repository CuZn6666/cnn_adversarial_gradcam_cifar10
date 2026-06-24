# WP3 Backward Manual Review Notes

## 1. Purpose of WP3 Manual Review

This note documents hand-computed examples for the WP3 manual backward
implementation.

The examples are grounded in deterministic tests from `tests/test_layers.py`.
The goal is to connect the code-level backward API to small computations that
can be checked by hand.

WP3 adds manual `backward(...)` methods on top of the WP2 forward-only layers.
The common API is:

```text
forward(inputs) stores the minimum cache needed for backward.
backward(grad_out) receives the upstream gradient.
backward(grad_out) returns grad_input.
Trainable layers also store grad_weight and grad_bias.
```

This note focuses on:

```text
Linear.backward
MaxPool2D.backward
Conv2D.backward
```

These are the most useful WP3 examples to review manually because they involve
explicit gradient formulas or gradient routing.

## 2. Linear backward hand-computed example

Relevant test in `tests/test_layers.py`:

```text
test_linear_backward_matches_hand_computed_gradients
```

The `Linear.forward(...)` formula is:

```text
out = inputs @ weights.T + bias
```

The test uses:

```python
inputs = [
    [1.0, 2.0],
    [3.0, 4.0],
]
```

with shape:

```text
(batch_size, in_features) = (2, 2)
```

The weights are:

```python
weights = [
    [ 1.0, 0.0],
    [ 0.0, 2.0],
    [-1.0, 1.0],
]
```

with shape:

```text
(out_features, in_features) = (3, 2)
```

The bias is:

```python
bias = [0.5, -1.0, 2.0]
```

with shape:

```text
(out_features,) = (3,)
```

The upstream gradient is:

```python
grad_out = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
]
```

with shape:

```text
(batch_size, out_features) = (2, 3)
```

The backward formulas are:

```text
grad_input = grad_out @ weights
grad_weight = grad_out.T @ inputs
grad_bias = grad_out.sum(axis=0)
```

### `grad_input`

For the first sample:

```text
[1.0, 2.0, 3.0] @ [
    [ 1.0, 0.0],
    [ 0.0, 2.0],
    [-1.0, 1.0],
]
```

First input feature:

```text
1.0 * 1.0 + 2.0 * 0.0 + 3.0 * (-1.0) = -2.0
```

Second input feature:

```text
1.0 * 0.0 + 2.0 * 2.0 + 3.0 * 1.0 = 7.0
```

For the second sample:

```text
4.0 * 1.0 + 5.0 * 0.0 + 6.0 * (-1.0) = -2.0
4.0 * 0.0 + 5.0 * 2.0 + 6.0 * 1.0 = 16.0
```

Therefore:

```python
grad_input = [
    [-2.0, 7.0],
    [-2.0, 16.0],
]
```

with shape:

```text
(2, 2)
```

### `grad_weight`

The formula is:

```text
grad_weight = grad_out.T @ inputs
```

For output feature 0:

```text
[1.0, 4.0] @ [
    [1.0, 2.0],
    [3.0, 4.0],
]
```

First weight:

```text
1.0 * 1.0 + 4.0 * 3.0 = 13.0
```

Second weight:

```text
1.0 * 2.0 + 4.0 * 4.0 = 18.0
```

For output feature 1:

```text
2.0 * 1.0 + 5.0 * 3.0 = 17.0
2.0 * 2.0 + 5.0 * 4.0 = 24.0
```

For output feature 2:

```text
3.0 * 1.0 + 6.0 * 3.0 = 21.0
3.0 * 2.0 + 6.0 * 4.0 = 30.0
```

Therefore:

```python
grad_weight = [
    [13.0, 18.0],
    [17.0, 24.0],
    [21.0, 30.0],
]
```

with shape:

```text
(3, 2)
```

### `grad_bias`

The formula is:

```text
grad_bias = grad_out.sum(axis=0)
```

Therefore:

```text
grad_bias[0] = 1.0 + 4.0 = 5.0
grad_bias[1] = 2.0 + 5.0 = 7.0
grad_bias[2] = 3.0 + 6.0 = 9.0
```

So:

```python
grad_bias = [5.0, 7.0, 9.0]
```

with shape:

```text
(3,)
```

## 3. MaxPool2D backward hand-computed example

Relevant test in `tests/test_layers.py`:

```text
test_max_pool2d_backward_routes_gradients_to_max_locations
```

The test creates:

```python
layer = MaxPool2D(kernel_size=2, stride=2)
```

The input tensor has shape `(1, 1, 4, 4)`:

```python
inputs = [
    [
        [
            [1.0, 3.0, 2.0, 4.0],
            [5.0, 0.0, 6.0, 1.0],
            [7.0, 8.0, 9.0, 2.0],
            [0.0, 1.0, 3.0, 10.0],
        ]
    ]
]
```

The upstream gradient has shape `(1, 1, 2, 2)`:

```python
grad_out = [
    [
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    ]
]
```

`MaxPool2D.backward(...)` routes each `grad_out` value only to the maximum
position in the corresponding pooling window.

Top-left window:

```python
[
    [1.0, 3.0],
    [5.0, 0.0],
]
```

The maximum is `5.0`, so it receives `grad_out[0, 0, 0, 0] = 1.0`.

Top-right window:

```python
[
    [2.0, 4.0],
    [6.0, 1.0],
]
```

The maximum is `6.0`, so it receives `grad_out[0, 0, 0, 1] = 2.0`.

Bottom-left window:

```python
[
    [7.0, 8.0],
    [0.0, 1.0],
]
```

The maximum is `8.0`, so it receives `grad_out[0, 0, 1, 0] = 3.0`.

Bottom-right window:

```python
[
    [9.0, 2.0],
    [3.0, 10.0],
]
```

The maximum is `10.0`, so it receives `grad_out[0, 0, 1, 1] = 4.0`.

All other positions receive `0.0`. Therefore:

```python
grad_input = [
    [
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    ]
]
```

with shape:

```text
(1, 1, 4, 4)
```

Related tie-handling test:

```text
test_max_pool2d_backward_routes_ties_to_first_row_major_maximum
```

For this input:

```python
inputs = [
    [
        [
            [5.0, 5.0],
            [1.0, 0.0],
        ]
    ]
]
```

and:

```python
grad_out = [[[[7.0]]]]
```

there are two equal maximum values. The implementation uses the first maximum
in row-major order, so the top-left `5.0` receives the gradient:

```python
grad_input = [
    [
        [
            [7.0, 0.0],
            [0.0, 0.0],
        ]
    ]
]
```

## 4. Conv2D backward hand-computed example

Relevant test in `tests/test_layers.py`:

```text
test_conv2d_backward_matches_hand_computed_gradients
```

The test creates:

```text
in_channels = 1
out_channels = 1
kernel_size = 2
padding = 0
stride = 1
```

The input tensor has shape `(1, 1, 3, 3)`:

```python
inputs = [
    [
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    ]
]
```

The weights have shape `(1, 1, 2, 2)`:

```python
weights = [
    [
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    ]
]
```

The bias is:

```python
bias = [0.0]
```

The upstream gradient has shape `(1, 1, 2, 2)`:

```python
grad_out = [
    [
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    ]
]
```

The project's `Conv2D.forward(...)` uses cross-correlation-style computation:
the kernel is not flipped during the forward pass. `Conv2D.backward(...)`
therefore accumulates gradients according to the exact local windows used by
that forward pass.

### `grad_bias`

The bias contributes once to every output position, so:

```text
grad_bias = 1.0 + 2.0 + 3.0 + 4.0 = 10.0
```

Therefore:

```python
grad_bias = [10.0]
```

### `grad_weight`

Each weight gradient is the sum of:

```text
corresponding input value * corresponding grad_out value
```

For the top-left kernel weight:

```text
1.0 * 1.0 + 2.0 * 2.0 + 4.0 * 3.0 + 5.0 * 4.0
= 1.0 + 4.0 + 12.0 + 20.0
= 37.0
```

For the top-right kernel weight:

```text
2.0 * 1.0 + 3.0 * 2.0 + 5.0 * 3.0 + 6.0 * 4.0
= 2.0 + 6.0 + 15.0 + 24.0
= 47.0
```

For the bottom-left kernel weight:

```text
4.0 * 1.0 + 5.0 * 2.0 + 7.0 * 3.0 + 8.0 * 4.0
= 4.0 + 10.0 + 21.0 + 32.0
= 67.0
```

For the bottom-right kernel weight:

```text
5.0 * 1.0 + 6.0 * 2.0 + 8.0 * 3.0 + 9.0 * 4.0
= 5.0 + 12.0 + 24.0 + 36.0
= 77.0
```

Therefore:

```python
grad_weight = [
    [
        [
            [37.0, 47.0],
            [67.0, 77.0],
        ]
    ]
]
```

### `grad_input`

Each input pixel receives contributions from every output window in which it
appears.

For example, the center input pixel `5.0` appears in all four output windows:

```text
top-left output uses weight 4.0 and grad_out 1.0
top-right output uses weight 3.0 and grad_out 2.0
bottom-left output uses weight 2.0 and grad_out 3.0
bottom-right output uses weight 1.0 and grad_out 4.0
```

So the center input gradient is:

```text
1.0 * 4.0 + 2.0 * 3.0 + 3.0 * 2.0 + 4.0 * 1.0
= 4.0 + 6.0 + 6.0 + 4.0
= 20.0
```

The full expected `grad_input` is:

```python
grad_input = [
    [
        [
            [1.0, 4.0, 4.0],
            [6.0, 20.0, 16.0],
            [9.0, 24.0, 16.0],
        ]
    ]
]
```

with shape:

```text
(1, 1, 3, 3)
```

## 5. Traceability table

| Component | Test name in `tests/test_layers.py` | Source implementation | What was manually checked |
| --- | --- | --- | --- |
| `Linear` | `test_linear_backward_matches_hand_computed_gradients` | `src/layers/forward.py::Linear.backward` | `grad_input`, `grad_weight`, and `grad_bias` formulas and shapes |
| `Linear` | `test_linear_backward_requires_forward_call` | `src/layers/forward.py::Linear.backward` | `RuntimeError` when `backward(...)` is called before `forward(...)` |
| `Linear` | `test_linear_backward_rejects_wrong_grad_out_shape` | `src/layers/forward.py::Linear.backward` | `ValueError` for incompatible `grad_out` shape |
| `MaxPool2D` | `test_max_pool2d_backward_routes_gradients_to_max_locations` | `src/layers/forward.py::MaxPool2D.backward` | Gradient routing to max positions only |
| `MaxPool2D` | `test_max_pool2d_backward_routes_ties_to_first_row_major_maximum` | `src/layers/forward.py::MaxPool2D.backward` | Deterministic first-maximum row-major tie behavior |
| `Conv2D` | `test_conv2d_backward_matches_hand_computed_gradients` | `src/layers/forward.py::Conv2D.backward` | `grad_input`, `grad_weight`, `grad_bias`, sliding-window accumulation |
| `CompactCNN` | `test_compact_cnn_backward_returns_input_gradient_shape` | `src/models/compact_cnn.py::CompactCNN.backward` | Reverse layer order and returned input-gradient shape |

## 6. Validation notes

This README is documentation only. It does not change source code, tests,
model behavior, PDFs, checkpoints, or experiment artifacts.

Useful validation commands are:

```bash
python -m pytest tests/test_layers.py -v
python -m pytest tests/test_backward.py -v
python -m pytest tests/test_losses.py -v
python -m pytest tests/test_integration.py -v
python -m pytest tests/ -v
```

These notes are not a replacement for automated tests. They are a manual review
aid for understanding how WP3 backward computations match the deterministic
test examples.
