# WP_PLAN.md

## Project Title

Compact CNN on CIFAR-10 with Adversarial Robustness and Grad-CAM Analysis

## Project Goal

The goal of this project is to implement and analyze a compact CNN on CIFAR-10, including:

1. project orientation and planning,
2. a clean data and configuration pipeline,
3. a compact CNN forward pass,
4. manual backward implementation,
5. gradient checking and input gradients,
6. baseline training and evaluation,
7. adversarial attacks such as FGSM or PGD,
8. Grad-CAM based visual explanations,
9. final experimental analysis and documentation.

## Current Status

| WP  | Title                                                                   | Status                    |
| --- | ----------------------------------------------------------------------- | ------------------------- |
| WP0 | Project orientation, literature/context preparation, and basic planning | mostly completed          |
| WP1 | Project setup and CIFAR-10 pipeline                                     | mostly completed          |
| WP2 | Compact CNN forward implementation                                      | completed                 |
| WP3 | Manual backward implementation                                          | not started / planned     |
| WP4 | Gradient check and input-gradient support                               | planned                   |
| WP5 | Baseline training and evaluation                                        | planned                   |
| WP6 | Adversarial attack implementation                                       | planned                   |
| WP7 | Grad-CAM implementation                                                 | planned                   |
| WP8 | Robustness and explanation analysis                                     | planned                   |
| WP9 | Final report and reproducibility cleanup                                | planned                   |

## Immediate Next Step

WP2 forward-only implementation and validation are completed.

WP3 manual backward implementation is the next planned Work Package, but it has
not been started.

## Work Package Details

### WP0: Project Orientation and Planning

Goal:

Understand the project goal, prepare the basic plan, inspect the provided materials, and define the overall implementation direction.

Expected deliverables:

* basic project understanding,
* initial Work Package plan,
* initial repository orientation,
* initial notes about later GPU/cluster usage.

Relevant folders/files:

```text
README.md
WP_PLAN.md
AGENTS.md
TESTING.md
docs/
```

Validation:

See `TESTING.md`, section WP0.

Status:

Mostly completed.

---

### WP1: Project Setup and CIFAR-10 Pipeline

Goal:

Set up the repository structure, dependencies, configuration files, and CIFAR-10 data loading pipeline.

Expected deliverables:

* clean project folder structure,
* dependency file such as `requirements.txt` or `pyproject.toml`,
* dataset loading code,
* basic README instructions,
* minimal tests or checks for data loading.

Relevant folders/files:

```text
README.md
requirements.txt
pyproject.toml
configs/
src/
src/data/
tests/
```

Validation:

See `TESTING.md`, section WP1.

Status:

Mostly completed.

---

### WP2: Compact CNN Forward Implementation

Goal:

Implement the compact CNN forward pass for CIFAR-10 classification.

Expected deliverables:

* compact CNN model structure,
* forward pass through convolution, activation, pooling, flattening, and linear classification layers,
* correct output shape for 10 CIFAR-10 classes,
* minimal forward tests,
* no training or backward implementation yet.

Relevant folders/files:

```text
src/
src/models/
src/layers/
tests/
configs/
```

Suggested implementation order:

1. Inspect the existing `src/` structure.
2. Decide where the compact CNN model should live.
3. Implement or complete the required forward layers.
4. Implement the compact CNN forward pass.
5. Add a minimal test with a fake CIFAR-10 batch.
6. Verify that the output shape is `(batch_size, 10)`.

Validation:

See `TESTING.md`, section WP2.

Status:

Completed.

---

### WP3: Manual Backward Implementation

Goal:

Implement the manual backward pass for the compact CNN.

Expected deliverables:

* backward pass for Linear layer,
* backward pass for ReLU,
* backward pass for MaxPool,
* backward pass for Conv2D,
* consistent gradient shapes,
* no NaN gradients,
* minimal tests for each backward component.

Relevant folders/files:

```text
src/
src/layers/
src/models/
tests/
```

Implementation order:

1. Linear backward
2. ReLU backward
3. MaxPool backward
4. Conv2D backward
5. Full model backward integration

Validation:

See `TESTING.md`, section WP3.

Status:

Not started / planned.

---

### WP4: Gradient Check and Input-Gradient Support

Goal:

Verify the manual backward implementation using numerical gradient checks and enable gradients with respect to the input image.

Expected deliverables:

* numerical gradient checking utilities,
* relative error comparison,
* input-gradient computation,
* tests for gradient correctness.

Relevant folders/files:

```text
src/
src/gradients/
tests/
experiments/
```

Validation:

See `TESTING.md`, section WP4.

Status:

Planned.

---

### WP5: Baseline Training and Evaluation

Goal:

Train the compact CNN on CIFAR-10 and evaluate baseline performance.

Expected deliverables:

* training script,
* evaluation script,
* saved metrics,
* loss and accuracy curves,
* reproducible configuration.

Relevant folders/files:

```text
configs/
experiments/
results/
src/
```

Validation:

See `TESTING.md`, section WP5.

Status:

Planned.

---

### WP6: Adversarial Attack Implementation

Goal:

Implement adversarial attacks based on input gradients, especially FGSM and optionally PGD.

Expected deliverables:

* FGSM implementation,
* optional PGD implementation,
* adversarial accuracy evaluation,
* epsilon-dependent robustness results.

Relevant folders/files:

```text
src/attacks/
experiments/
configs/
results/
```

Validation:

See `TESTING.md`, section WP6.

Status:

Planned.

---

### WP7: Grad-CAM Implementation

Goal:

Implement Grad-CAM visual explanations for the compact CNN.

Expected deliverables:

* Grad-CAM computation,
* heatmap generation,
* visualization script,
* example visualizations on clean and adversarial images.

Relevant folders/files:

```text
src/explainability/
experiments/
results/
deliverables/
```

Validation:

See `TESTING.md`, section WP7.

Status:

Planned.

---

### WP8: Robustness and Explanation Analysis

Goal:

Compare clean and adversarial behavior using accuracy, confidence, and Grad-CAM visualizations.

Expected deliverables:

* comparison tables,
* plots,
* selected visual examples,
* written interpretation.

Relevant folders/files:

```text
experiments/
results/
deliverables/
```

Validation:

See `TESTING.md`, section WP8.

Status:

Planned.

---

### WP9: Final Report and Reproducibility Cleanup

Goal:

Prepare final documentation and make the project reproducible.

Expected deliverables:

* final README,
* clear run instructions,
* final plots,
* final result summary,
* clean project structure,
* optional cluster instructions.

Relevant folders/files:

```text
README.md
deliverables/
results/
scripts/
docs/
```

Validation:

See `TESTING.md`, section WP9.

Status:

Planned.

## Rule for Moving Between Work Packages

Before moving from one WP to the next:

1. Check the expected deliverables.
2. Run the relevant tests in `TESTING.md`.
3. Fix blocking failures.
4. Update the status table above.
5. Make a Git commit if the repository is in a stable state.
