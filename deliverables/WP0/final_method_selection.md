# WP0 Final Method Selection

## Selection Basis

This document records the method selection that is consistent with the current
repository state. It does not claim that every originally planned method has
been implemented.

## Selected and Implemented

| Area | Selected Method | Current Status | Repository Evidence |
| --- | --- | --- | --- |
| Reference model | Compact CNN implemented from scratch with NumPy | Implemented | `src/models/compact_cnn.py`, `src/layers/forward.py` |
| Loss | Softmax Cross-Entropy | Implemented | `src/losses/cross_entropy.py` |
| Optimizer | SGD | Implemented | `src/optimizers/sgd.py` |
| Gradient validation | Finite-difference checks for selected components | Implemented | `tests/test_gradient_check.py` |
| White-box attack | FGSM | Implemented | `src/attacks/fgsm.py`, `src/robustness.py` |
| Explainability | Grad-CAM on `relu2` activation | Implemented, needs formal WP13 closeout | `src/gradcam.py`, `tests/test_gradcam.py` |

## Selected but Deferred

| Area | Selected Method | Current Status | Reason |
| --- | --- | --- | --- |
| Stronger white-box attack | PGD | Deferred | Current active phase prioritizes CuPy acceleration and larger FGSM evaluation. |
| Non-gradient attack | Simplified square-based black-box random search | Deferred | Current active phase prioritizes the validated FGSM pipeline and GPU scaling. |
| Query-count evaluation | Query-efficiency metrics for black-box attack | Deferred | Depends on the deferred black-box implementation. |

## Active Development Priority

The active development sequence is:

```text
NumPy reference
-> CuPy acceleration
-> CPU/GPU numerical equivalence
-> cluster execution
-> large-scale FGSM evaluation
-> runtime/robustness analysis
```

PGD and black-box attacks should not be implemented during this cycle unless
the project direction is explicitly changed.

## Remaining TODO

If this document is used as a formal report appendix, add citations for FGSM,
PGD, Square Attack-style black-box methods, and Grad-CAM.
