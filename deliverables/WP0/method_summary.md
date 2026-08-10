# WP0 Method Summary

## Scope

This document summarizes the method choices reflected by the current
repository implementation. It is based on repository evidence, not on a new
literature review.

## Current Implemented Methods

### CompactCNN Reference Model

The project uses a compact CNN implemented from scratch with NumPy as the
reference backend. The implemented architecture is:

```text
Input (N, 3, 32, 32)
-> Conv2D(3 -> 8, kernel_size=3, padding=1)
-> ReLU
-> MaxPool2D(kernel_size=2, stride=2)
-> Conv2D(8 -> 16, kernel_size=3, padding=1)
-> ReLU
-> MaxPool2D(kernel_size=2, stride=2)
-> Flatten
-> Linear(16 * 8 * 8 -> 10)
-> logits
```

The forward and backward passes are implemented manually for the project
layers. `SoftmaxCrossEntropyLoss` and SGD are also implemented directly.

### Gradient Validation

Manual gradients are checked with deterministic unit tests and selected
finite-difference numerical gradient checks. The tested components include
`Linear`, `Conv2D`, `SoftmaxCrossEntropyLoss`, and the full
loss-to-input-gradient pipeline.

### FGSM

FGSM is the implemented white-box adversarial attack. It uses gradients of the
loss with respect to input images:

```python
np.clip(images + epsilon * np.sign(grad_input), 0.0, 1.0)
```

The repository includes qualitative FGSM examples, batched FGSM robustness
evaluation, epsilon sweeps, attack success-rate aggregation, and a 1024-sample
quantitative evaluation.

### Grad-CAM

Grad-CAM is implemented for qualitative explainability. The current target
activation is the `relu2` output before `pool2`. The repository includes clean
and FGSM-adversarial Grad-CAM comparisons.

## Deferred Methods

The original plan included PGD and one simplified square-based black-box
attack. These methods remain intentionally deferred. They are not part of the
current active development cycle, which prioritizes CuPy acceleration and
larger FGSM evaluation.

## Current Active Extension Direction

The next active technical direction is:

```text
NumPy reference
-> CuPy acceleration
-> CPU/GPU numerical equivalence
-> cluster execution
-> large-scale FGSM evaluation
-> runtime/robustness analysis
```

## Remaining TODO

Add exact external literature citations if this deliverable is used directly
in a final report.
