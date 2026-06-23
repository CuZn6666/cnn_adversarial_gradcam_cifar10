# WP2 Manual Review Notes

This directory contains handwritten scanned manual review notes for WP2.

> **PDF rendering note:** Due to rendering issues, the uploaded PDF files may not be previewable on GitHub, but they can be downloaded and viewed locally.

The notes document manual study and validation of the WP2 compact CNN forward implementation, including:

- CompactCNN forward structure
- sliding windows and local patch extraction
- convolution-related manual understanding
- WP2 code comprehension and manual validation

These files are documentation artifacts only. They do not affect source code, tests, or experiment outputs.

## WP2 Forward Manual Review Notes

## 1. test_conv2d_forward_matches_hand_computed_output

This test checks a minimal Conv2D example.

Parameters:

in_channels=1
out_channels=1
kernel_size=2

Manually set:

weights = [[[[1.0, 0.0],
[0.0, -1.0]]]]

bias = [0.5]

The input is:

[[1, 2, 3],
[4, 5, 6],
[7, 8, 9]]

Because the kernel is 2x2 and the default stride should be 1, the output is 2x2.

Manual computation for each window:

Top-left window:

[[1, 2],
[4, 5]]

Convolution result:

1*1 + 2*0 + 4*0 + 5*(-1) + 0.5
= 1 - 5 + 0.5
= -3.5

Top-right window:

[[2, 3],
[5, 6]]

2*1 + 3*0 + 5*0 + 6*(-1) + 0.5
= 2 - 6 + 0.5
= -3.5

Bottom-left window:

[[4, 5],
[7, 8]]

4 - 8 + 0.5 = -3.5

Bottom-right window:

[[5, 6],
[8, 9]]

5 - 9 + 0.5 = -3.5

Therefore, the expected output is:

[[[[-3.5, -3.5],
[-3.5, -3.5]]]]

The shape is:

(1, 1, 2, 2)

---

## 2. test_relu_forward_matches_hand_computed_output

Input:

[[-2.0, -0.5, 0.0, 3.0]]

ReLU rule:

ReLU(x) = max(0, x)

Therefore:

-2.0  -> 0.0
-0.5  -> 0.0
0.0  -> 0.0
3.0  -> 3.0

Expected:

[[0.0, 0.0, 0.0, 3.0]]

Note that the output for 0.0 is still 0.0.

---

## 3. test_max_pool2d_forward_matches_hand_computed_output

The input comes from:

np.arange(1, 17).reshape(1, 1, 4, 4)

That is:

[[ 1,  2,  3,  4],
[ 5,  6,  7,  8],
[ 9, 10, 11, 12],
[13, 14, 15, 16]]

MaxPool2D(kernel_size=2, stride=2) means that each operation looks at a 2x2 region and moves without overlap.

Four windows:

Top-left:

[[1, 2],
[5, 6]]
max = 6

Top-right:

[[3, 4],
[7, 8]]
max = 8

Bottom-left:

[[ 9, 10],
[13, 14]]
max = 14

Bottom-right:

[[11, 12],
[15, 16]]
max = 16

Therefore, the expected output is:

[[[[6.0, 8.0],
[14.0, 16.0]]]]

The shape is:

(1, 1, 2, 2)

---

## 4. test_flatten_forward_preserves_batch_order

Input:

np.arange(16).reshape(2, 2, 2, 2)

The shape is:

(N, C, H, W) = (2, 2, 2, 2)

This means that the batch contains 2 samples.

First sample:

channel 0:
[[0, 1],
[2, 3]]

channel 1:
[[4, 5],
[6, 7]]

After flattening:

[0, 1, 2, 3, 4, 5, 6, 7]

Second sample:

channel 0:
[[ 8,  9],
[10, 11]]

channel 1:
[[12, 13],
[14, 15]]

After flattening:

[8, 9, 10, 11, 12, 13, 14, 15]

Therefore, the expected output is:

[
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
]

The core purpose of this test is not complex computation, but confirming that:

the batch dimension is not mixed up
each sample is flattened independently
the flattening order is the C -> H -> W row-major order

---

## 5. test_linear_forward_matches_hand_computed_output

The formula for Linear.forward(x) is:

out = x @ weights.T + bias

Here:

weights = [
[ 1.0, 0.0],
[ 0.0, 2.0],
[-1.0, 1.0],
]

bias = [0.5, -1.0, 2.0]

Input:

inputs = [
[1.0, 2.0],
[3.0, 4.0],
]

First sample [1, 2]:

First output:

1*1 + 2*0 + 0.5 = 1.5

Second output:

1*0 + 2*2 - 1.0 = 3.0

Third output:

1*(-1) + 2*1 + 2.0 = 3.0

Therefore, the first row is:

[1.5, 3.0, 3.0]

Second sample [3, 4]:

First output:

3*1 + 4*0 + 0.5 = 3.5

Second output:

3*0 + 4*2 - 1.0 = 7.0

Third output:

3*(-1) + 4*1 + 2.0 = 3.0

Therefore, the second row is:

[3.5, 7.0, 3.0]

Expected:

[
[1.5, 3.0, 3.0],
[3.5, 7.0, 3.0],
]

---

## 6. test_compact_cnn_forward_shape_and_finite_output

This test does not manually compute concrete numerical values. Instead, it verifies the basic output properties of CompactCNN.forward.

It uses:

@pytest.mark.parametrize("batch_size", [1, 2])

Therefore, it tests both:

batch_size = 1
batch_size = 2

Input shape:

(batch_size, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)

For CIFAR-10, this is:

(batch_size, 3, 32, 32)

The test requires the output to be:

outputs.shape == (batch_size, 10)

This is because CIFAR-10 has 10 classes.

It also checks:

np.isfinite(outputs).all()

This means the output must not contain:

NaN
inf
-inf

---

## 7. test_compact_cnn_fixed_seed_is_reproducible

This test verifies that the model forward pass is reproducible when a fixed seed is used.

The input is fixed:

inputs = np.random.default_rng(7).random(...)

Then two models are created:

CompactCNN(seed=42)
CompactCNN(seed=42)

Because the seed is the same, the initialized parameters of the two models should be exactly the same.

Therefore:

first_outputs = CompactCNN(seed=42).forward(inputs)
second_outputs = CompactCNN(seed=42).forward(inputs)

should be exactly identical.

The test uses:

np.testing.assert_array_equal(first_outputs, second_outputs)

This is stricter than assert_allclose because it requires the arrays to be exactly equal element by element.

---

## 8. test_compact_cnn_rejects_invalid_input_shape

Here, an intentionally wrong shape is passed:

invalid_inputs = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

That is:

(1, 32, 32, 3)

This is the NHWC format.

However, the model in this project expects:

(N, C, H, W)

That is:

(1, 3, 32, 32)

Therefore, the test expects CompactCNN.forward to raise:

ValueError

and the error message should match:

CompactCNN expects input with shape (N, 3, 32, 32).

---

In summary, test_forward.py covers the following forward behaviors:

Conv2D    -> manually computed convolution values
ReLU      -> manually computed max(0, x)
MaxPool2D -> manually computed 2x2 max pooling
Flatten   -> checks batch order and flattening order
Linear    -> manually computed x @ weights.T + bias
CompactCNN -> checks output shape, finite values, seed reproducibility, and invalid input shape handling
