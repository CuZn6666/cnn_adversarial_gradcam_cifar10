# WP_PLAN.md

## Project Title

Compact CNN on CIFAR-10 with Adversarial Robustness and Grad-CAM Analysis

## Project Goal

The goal of this project is to implement and analyze a compact CNN on CIFAR-10
through the complete sequence defined in `WorkPackagePlan.txt`:

1. focused literature review and final method selection,
2. project setup and the CIFAR-10 data pipeline,
3. compact CNN forward implementation,
4. manual backward implementation,
5. gradient checks and input-gradient support,
6. baseline training and clean evaluation,
7. focused runtime bottleneck handling,
8. FGSM implementation, visualization, and robustness evaluation,
9. PGD implementation, evaluation, and comparison,
10. non-gradient black-box attack implementation and evaluation,
11. Grad-CAM implementation and qualitative attack analysis,
12. final integration, reproducibility, and result organization.

The source spreadsheet estimates a total technical workload of 360 hours
(180 hours per project participant), excluding the final report and poster.
Individual estimates are planning values and may vary during implementation.

## Current Status

| WP   | Title                                                    | Status           |
| ---- | -------------------------------------------------------- | ---------------- |
| WP0  | Focused Literature Review and Final Method Selection     | mostly completed |
| WP1  | Project setup and CIFAR-10 pipeline                      | mostly completed |
| WP2  | Compact CNN forward implementation                       | completed        |
| WP3  | Manual Backward Implementation                           | in progress      |
| WP4  | Gradient Checks and Input-Gradient Support               | planned          |
| WP5  | Baseline training and clean evaluation                   | planned          |
| WP6  | Focused Runtime Bottleneck Handling                      | planned          |
| WP7  | FGSM Attack and Input-Gradient Visualization             | planned          |
| WP8  | FGSM robustness evaluation                               | planned          |
| WP9  | PGD Attack Implementation                                | planned          |
| WP10 | PGD Robustness Evaluation and Comparison                 | planned          |
| WP11 | Non-Gradient Black-Box Attack Implementation             | planned          |
| WP12 | Black-Box Attack Evaluation                              | planned          |
| WP13 | Grad-CAM Implementation                                  | planned          |
| WP14 | Grad-CAM Analysis Before and After Attacks               | planned          |
| WP15 | Final integration, reproducibility and result organization | planned        |

## Immediate Next Step

WP2 forward implementation and validation are completed.

WP3 manual backward implementation is in progress. `Linear.backward` is
implemented and tested. The next implementation step is `ReLU.backward`.

WP4–WP15 remain planned. Training, runtime optimization, adversarial attacks,
Grad-CAM, and final integration must not start before their prerequisite Work
Packages are completed and validated.

## Work Package Details

### WP0: Focused Literature Review and Final Method Selection

Goal:

Review Lecture 04 Example 8 and conduct focused literature research on
adversarial attacks and explainability. Select suitable gradient-based methods,
including FGSM and PGD, and one non-gradient-based black-box method, such as a
simplified square-based attack. Define the final evaluation metrics, including
accuracy drop, attack success rate, query count, and qualitative Grad-CAM
analysis.

Expected deliverables:

* short method summary,
* final attack selection,
* final evaluation metrics,
* updated project scope.

Relevant folders/files:

```text
README.md
AGENTS.md
WP_PLAN.md
TESTING.md
WorkPackagePlan.txt
WorkPackagePlan.xlsx
deliverables/WP0/project_scope_draft.md
deliverables/WP0/method_summary.md
deliverables/WP0/final_method_selection.md
deliverables/WP0/evaluation_metrics.md
```

Suggested implementation order:

1. Review Lecture 04 Example 8 and related examples — 2h.
2. Review introductory FGSM and adversarial-example references — 3h.
3. Review introductory PGD and robust-optimization references — 3h.
4. Review the Grad-CAM reference — 3h.
5. Review non-gradient and black-box attack references — 4h.
6. Decide the final methods and evaluation metrics — 3h.

Validation:

* The project scope, selected attacks, explainability method, and evaluation
  metrics are documented consistently.
* The WP0 deliverables contain the method summary, final attack selection,
  evaluation metrics, and updated scope.
* `AGENTS.md`, `WP_PLAN.md`, and `TESTING.md` exist and agree on the active Work
  Package.

Suggested checks:

```bash
ls AGENTS.md WP_PLAN.md TESTING.md
ls deliverables/WP0/
```

Dependencies:

* No earlier Work Package dependency.
* Requires access to the project brief, Lecture 04 Example 8, and the focused
  literature references used for method selection.

Estimated duration:

18h total (9h per participant).

Status:

Mostly completed.

---

### WP1: Project setup and CIFAR-10 pipeline

Goal:

Set up the project structure, environment, data folders, logging structure, and
reproducibility settings. Implement CIFAR-10 loading, preprocessing, batching,
and basic dataset utilities. Keep the infrastructure compact and focused on the
needs of later experiments.

Expected deliverables:

* working CIFAR-10 data pipeline,
* reproducible project skeleton.

Relevant folders/files:

```text
requirements.txt
configs/default_config.py
src/data/__init__.py
src/data/cifar10_loader.py
src/data/batching.py
src/utils/seed.py
experiments/check_data_pipeline.py
tests/test_data_pipeline.py
results/checkpoints/.gitkeep
results/figures/
deliverables/WP1/
```

Suggested implementation order:

1. Set up the project structure and environment — 4h.
2. Implement CIFAR-10 loading and preprocessing — 4h.
3. Implement mini-batch utilities — 3h.
4. Set up logging and result directories — 3h.
5. Add basic reproducibility settings — 2h.

Validation:

* CIFAR-10 training and test arrays have the expected NCHW shapes.
* Images use `float32` values in `[0, 1]`.
* Labels use valid class indices from `0` to `9`.
* Mini-batches have the expected image and label shapes.
* A fixed seed reproduces the same shuffled first batch.
* The data pipeline sanity-check script runs without dataset, path, or checksum
  errors.

Suggested commands:

```bash
.venv/bin/python -m pytest tests/test_data_pipeline.py -v
.venv/bin/python -m experiments.check_data_pipeline
```

Dependencies:

* WP0 project scope and method selection are sufficiently defined.
* The Python environment can install the versions listed in
  `requirements.txt`.
* The CIFAR-10 archive is locally available or can be downloaded from the
  configured official URL.

Estimated duration:

16h total (8h per participant).

Status:

Mostly completed.

---

### WP2: Compact CNN forward implementation

Goal:

Implement the forward pass of a compact CNN from scratch. This includes
`Conv2D`, `ReLU`, `MaxPool`, `Flatten`, `Linear`, and Softmax Cross-Entropy
loss. The implementation is limited to the architecture required by this
project rather than a general deep learning framework.

Expected deliverables:

* compact CNN forward pass,
* shape tests,
* initial predictions on CIFAR-10 batches.

Relevant folders/files:

```text
configs/default_config.py
src/layers/__init__.py
src/layers/forward.py
src/models/__init__.py
src/models/compact_cnn.py
tests/test_forward.py
```

Suggested implementation order:

1. Implement `Conv2D` forward — 9h.
2. Implement `ReLU` and pooling forward — 5h.
3. Implement `Flatten` and `Linear` — 4h.
4. Implement Softmax Cross-Entropy loss — 4h.
5. Add shape tests and debug the forward pipeline — 6h.

Validation:

* Deterministic forward tests pass for `Conv2D`, `ReLU`, `MaxPool2D`,
  `Flatten`, and `Linear`.
* `CompactCNN.forward` accepts CIFAR-10 NCHW inputs with shape
  `(batch_size, 3, 32, 32)`.
* Model output has shape `(batch_size, 10)` for tested batch sizes.
* Forward outputs are finite and fixed-seed initialization is reproducible.
* Invalid model input shapes are rejected.
* Existing data-pipeline tests continue to pass.

Suggested commands:

```bash
.venv/bin/python -m pytest tests/test_forward.py -v
.venv/bin/python -m pytest tests/ -v
```

Known scope gap:

* The source plan includes Softmax Cross-Entropy forward in WP2, but no
  corresponding loss implementation exists in the current repository. Its
  file location and validation remain `TBD`.

Dependencies:

* WP1 data pipeline and the NCHW input convention are implemented and
  validated.
* `NUM_CLASSES`, image dimensions, channel count, and reproducibility seed are
  defined in `configs/default_config.py`.

Estimated duration:

28h total (14h per participant).

Status:

Completed.

---

### WP3: Manual Backward Implementation

Goal:

Implement the manual backward pass for the CNN layers. This includes gradients
for `Linear`, `ReLU`, `MaxPool`, `Conv2D`, and Softmax Cross-Entropy loss. The
backward pipeline forms the technical basis for training and for computing
gradients with respect to input images and feature maps.

Expected deliverables:

* manual backward pass for the compact CNN.

Relevant folders/files:

```text
src/layers/forward.py
src/layers/__init__.py
src/models/compact_cnn.py
src/models/__init__.py
tests/test_layers.py
tests/test_backward.py          # planned; not created yet
TBD                             # Softmax Cross-Entropy implementation location
```

Suggested implementation order:

1. Implement `Linear.backward` — 5h — completed and tested.
2. Implement `ReLU.backward` — 4h — next step.
3. Implement `MaxPool.backward` — 6h.
4. Implement `Conv2D.backward` — 12h.
5. Implement Softmax Cross-Entropy backward — 3h.

Validation:

* Each backward method requires an appropriate preceding forward call.
* `Linear.backward`, `ReLU.backward`, `MaxPool.backward`, and
  `Conv2D.backward` return gradients with the expected shapes.
* Parameterized layers store parameter gradients with shapes matching their
  parameters.
* Deterministic hand-computed tests validate the backward values for each
  layer.
* Max-pooling gradients are routed to the selected maximum locations.
* The full model backward pipeline runs after layer-level backward components
  are validated.
* All produced gradients are finite.
* Existing data-pipeline and forward tests continue to pass.
* Softmax Cross-Entropy backward validation remains `TBD` until its forward
  implementation and file location are defined.

Suggested commands:

```bash
.venv/bin/python -m pytest tests/test_layers.py -v
.venv/bin/python -m pytest tests/test_backward.py -v
.venv/bin/python -m pytest tests/ -v
```

Run `tests/test_backward.py` after full model backward integration exists.

Dependencies:

* WP2 forward layers and `CompactCNN.forward` are implemented and validated.
* Forward methods must cache only the values required by their corresponding
  backward methods.
* Softmax Cross-Entropy forward from the source-defined WP2 scope is required
  before implementing its backward pass; this dependency is currently
  unresolved.
* Numerical gradient checking is intentionally deferred to WP4.

Estimated duration:

30h total (15h per participant).

Status:

In progress. `Linear.backward` is completed and tested; `ReLU.backward` is next.

---

### WP4: Gradient Checks and Input-Gradient Support

Goal:

Verify selected gradients numerically to ensure that manual backpropagation is
correct. Extend the backward pipeline so that gradients with respect to input
images can be computed. This support is required by FGSM, PGD, and
input-gradient visualization.

Expected deliverables:

* selected gradient-check results,
* input-gradient computation support.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Add a numerical gradient check for the linear layer — 3h.
2. Add a numerical gradient check for `Conv2D` — 5h.
3. Add a numerical gradient check for the loss function — 3h.
4. Implement gradients with respect to input images — 4h.
5. Debug and document gradient correctness — 3h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

18h total (9h per participant).

Status:

Planned.

---

### WP5: Baseline training and clean evaluation

Goal:

Implement the training loop, optimizer updates, clean test evaluation, and
result logging. Train the compact CNN on CIFAR-10 and generate baseline loss
and accuracy curves. The trained model will be the target for adversarial
attacks and Grad-CAM analysis.

Expected deliverables:

* trained baseline CNN,
* clean accuracy,
* loss and accuracy curves.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Implement optimizer and parameter updates — 5h.
2. Implement the training loop — 7h.
3. Implement clean test evaluation — 4h.
4. Generate loss and accuracy plots — 4h.
5. Debug and stabilize baseline training — 6h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

26h total (13h per participant).

Status:

Planned.

---

### WP6: Focused Runtime Bottleneck Handling

Goal:

Identify the most expensive implementation components, such as convolution
forward or backward computation. Select one optimization or backend path, for
example CuPy or a selected vectorized implementation. The goal is to understand
and improve the main bottleneck rather than compare many frameworks.

Expected deliverables:

* one selected implementation path rather than a broad NumPy, CuPy, JAX, and
  PyTorch comparison,
* runtime measurements focused on understanding computational bottlenecks,
* short bottleneck discussion.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Identify computational bottlenecks — 3h.
2. Choose one implementation path — 2h.
3. Optimize the selected operation or backend path — 8h.
4. Perform basic runtime measurements — 3h.
5. Write a short bottleneck discussion — 2h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

18h total (9h per participant).

Status:

Planned.

---

### WP7: FGSM Attack and Input-Gradient Visualization

Goal:

Implement the first gradient-based adversarial attack using FGSM. Compute
gradients of the loss with respect to input images, visualize those gradients,
and use them to generate adversarial examples in the style of Lecture 04
Example 8.

Expected deliverables:

* input-gradient maps,
* FGSM adversarial examples,
* perturbation visualizations.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Compute loss gradients with respect to input images — 5h.
2. Visualize input gradients — 5h.
3. Implement the FGSM attack — 8h.
4. Generate adversarial examples for selected epsilon values — 7h.
5. Save adversarial examples and perturbation maps — 5h.
6. Debug and validate FGSM behavior — 4h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

34h total (17h per participant).

Status:

Planned.

---

### WP8: FGSM robustness evaluation

Goal:

Evaluate the trained CNN under FGSM attacks with different perturbation
strengths. Compare clean and adversarial accuracy, plot accuracy against epsilon
values, and select representative successful and failed attacks.

Expected deliverables:

* clean-versus-FGSM accuracy table,
* accuracy-versus-epsilon plot,
* representative FGSM examples.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Evaluate clean versus FGSM accuracy — 5h.
2. Test multiple epsilon values — 5h.
3. Plot accuracy against epsilon — 4h.
4. Select representative successful and failed attacks — 2h.
5. Summarize FGSM findings — 2h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

18h total (9h per participant).

Status:

Planned.

---

### WP9: PGD Attack Implementation

Goal:

Implement PGD as a stronger iterative gradient-based white-box attack. Include
the iterative attack loop, projection back to the allowed epsilon-ball,
clipping to the valid image range, and parameter handling for step size and
iteration count.

Expected deliverables:

* PGD attack implementation,
* working adversarial examples from the iterative attack.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Implement the iterative attack loop — 8h.
2. Implement projection to the epsilon-ball — 6h.
3. Add clipping to the valid image range — 4h.
4. Add parameter handling for step size and number of steps — 4h.
5. Debug PGD on selected examples — 4h.
6. Document differences from FGSM — 2h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

28h total (14h per participant).

Status:

Planned.

---

### WP10: PGD Robustness Evaluation and Comparison

Goal:

Evaluate PGD on a selected test subset or the full test set, depending on
runtime. Compare FGSM and PGD in terms of accuracy drop and attack strength to
obtain a quantitative comparison between one-step and iterative
gradient-based attacks.

Expected deliverables:

* FGSM-versus-PGD comparison,
* robustness curves,
* short gradient-based attack discussion.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Evaluate PGD on a selected test subset or the full test set — 5h.
2. Compare FGSM and PGD accuracy drop — 4h.
3. Plot robustness curves — 3h.
4. Select representative PGD examples — 2h.
5. Summarize the gradient-based attack comparison — 2h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

16h total (8h per participant).

Status:

Planned. This Work Package covers small-scale PGD evaluation, not large-scale
adversarial training.

---

### WP11: Non-Gradient Black-Box Attack Implementation

Goal:

Implement one non-gradient-based black-box adversarial attack. The selected
candidate in the source plan is a simplified square-based random-search attack
that perturbs random image regions and keeps changes that reduce model
confidence or increase loss.

Expected deliverables:

* non-gradient black-box attack implementation,
* black-box adversarial examples.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Select the final non-gradient method from the literature — 4h.
2. Define the black-box access setting — 3h.
3. Implement square-based random perturbation search — 10h.
4. Implement the score-based accept/reject criterion — 5h.
5. Define the query budget and stopping criteria — 4h.
6. Generate black-box adversarial examples — 4h.
7. Debug and validate attack behavior — 2h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

32h total (16h per participant).

Status:

Planned.

---

### WP12: Black-Box Attack Evaluation

Goal:

Evaluate the non-gradient-based attack using attack success rate and average
model-query count. Compare its behavior qualitatively with FGSM and PGD while
noting that it does not use internal gradients. Optionally test a small
one-pixel-style experiment if time permits.

Expected deliverables:

* attack success rate,
* average query count,
* black-box attack examples,
* short comparison with gradient-based attacks.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Measure attack success rate — 5h.
2. Measure the average number of queries — 4h.
3. Compare the black-box attack qualitatively with FGSM and PGD — 4h.
4. Select representative black-box examples — 3h.
5. Summarize limitations of the non-gradient attack — duration TBD; the source
   lists `2h---4h`.
6. Optionally run a small one-pixel attack test if time permits — 2h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

20h total (10h per participant).

Status:

Planned.

---

### WP13: Grad-CAM Implementation

Goal:

Implement Grad-CAM as a qualitative explainability tool. Store the final
convolutional feature maps, compute target-class gradients with respect to
those maps, calculate channel weights using global average pooling, and
generate normalized heatmaps that can be resized and overlaid on CIFAR-10
images.

Expected deliverables:

* Grad-CAM heatmaps,
* Grad-CAM overlays for selected images.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Store the final convolutional feature maps — 4h.
2. Compute target-class gradients with respect to feature maps — 6h.
3. Compute channel weights using global average pooling — 4h.
4. Generate and normalize Grad-CAM heatmaps — 5h.
5. Resize and overlay heatmaps on CIFAR-10 images — 5h.
6. Test Grad-CAM on selected examples — 4h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

28h total (14h per participant).

Status:

Planned.

---

### WP14: Grad-CAM Analysis Before and After Attacks

Goal:

Use Grad-CAM to inspect how the model's focus changes before and after
adversarial perturbations. Analyze correctly classified clean examples,
misclassified clean examples, and adversarial examples generated by FGSM, PGD,
and the black-box attack. Use Grad-CAM qualitatively rather than as a
quantitative model-comparison metric.

Expected deliverables:

* clean-versus-adversarial Grad-CAM comparison,
* qualitative case studies,
* visualization figures for the final report or poster.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Generate Grad-CAM for correctly classified clean examples — 5h.
2. Generate Grad-CAM for misclassified clean examples — 4h.
3. Generate Grad-CAM after FGSM and PGD attacks — 6h.
4. Generate Grad-CAM after the black-box attack — 5h.
5. Compare focus regions before and after attacks — 4h.
6. Prepare qualitative visualization figures — 4h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

28h total (14h per participant).

Status:

Planned.

---

### WP15: Final integration, reproducibility and result organization

Goal:

Integrate all components into a reproducible project package, preferably using
the Git repository. Organize final plots, analysis results, accuracy tables,
attack success-rate summaries, query-count summaries, Grad-CAM visualizations,
and README instructions. Perform final reproducibility checks and document the
limitations of the implemented methods.

Expected deliverables:

* complete reproducible codebase,
* final evaluation tables,
* final visualizations,
* robustness and explainability summary.

Relevant folders/files:

```text
TBD — not specified in source spreadsheet.
```

Suggested implementation order:

1. Prepare final clean and attacked accuracy tables — 5h.
2. Prepare the attack success-rate summary — 4h.
3. Prepare the query-efficiency summary for the black-box attack — 4h.
4. Prepare final plots and visualizations — 5h.
5. Clean up code and integrate modules — 4h.
6. Complete README and run instructions — 3h.
7. Perform reproducibility checks — 3h.
8. Write the final limitations discussion — 2h.

Validation:

TBD — not specified in source spreadsheet.

Dependencies:

TBD — not specified in source spreadsheet.

Estimated duration:

30h total (15h per participant).

Status:

Planned.

## Rule for Moving Between Work Packages

Before moving from one Work Package to the next:

1. Check the source-defined goal, subtasks, and expected deliverables.
2. Define or confirm validation criteria when the source spreadsheet lists
   them as `TBD`.
3. Run the relevant tests or reproducibility checks.
4. Fix blocking failures before proceeding.
5. Update the status table and the detailed Work Package status.
6. Make a Git commit when the repository is in a stable state.
7. Do not start later training, attack, explainability, or integration Work
   Packages before their prerequisites are implemented and validated.
