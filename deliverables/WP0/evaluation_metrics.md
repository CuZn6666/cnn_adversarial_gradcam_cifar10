# WP0 Evaluation Metrics

## Metrics Used in the Current Repository

### Clean Accuracy

Clean accuracy is the fraction of examples whose clean prediction equals the
ground-truth label:

```text
clean_accuracy = clean_correct / total_samples
```

It is used in baseline evaluation and FGSM robustness evaluation.

### Adversarial Accuracy

Adversarial accuracy is the fraction of examples whose adversarial prediction
equals the ground-truth label:

```text
adversarial_accuracy = adversarial_correct / total_samples
```

### Accuracy Drop

Accuracy drop measures the decrease from clean to adversarial accuracy:

```text
accuracy_drop = clean_accuracy - adversarial_accuracy
```

### Attack Success Rate

Attack success rate is defined only over clean-correct examples:

```text
attack_success_rate = successful_attacks / clean_correct_samples
```

A successful FGSM attack is a sample where the clean prediction is correct and
the adversarial prediction is incorrect.

When there are no clean-correct samples, the current implementation reports
`0.0` for attack success rate.

### Runtime

Runtime measurements are used for focused engineering analysis. WP6 recorded
`Conv2D.forward`, `Conv2D.backward`, and `train_step` timing. Future extension
work should record CPU/GPU runtime, sample count, batch size, epsilon count,
backend, and device.

### Confusion Matrix

The stronger reproducible baseline runner generates a raw-count CIFAR-10
confusion matrix for the selected test subset.

### Grad-CAM Qualitative Analysis

Grad-CAM is used qualitatively to inspect localization patterns before and
after FGSM. It is not used as a quantitative model-comparison metric.

## Metrics Planned but Deferred

| Metric | Depends On | Current Status |
| --- | --- | --- |
| PGD adversarial accuracy | PGD implementation | Deferred |
| PGD attack success rate | PGD implementation | Deferred |
| Black-box attack success rate | Black-box attack implementation | Deferred |
| Average query count | Black-box attack implementation | Deferred |

## Current Result Evidence

The repository currently records:

* baseline metrics in `results/baseline/portfolio_final_metrics.json`,
* FGSM quantitative metrics in `results/fgsm/fgsm_quantitative_metrics.json`,
* historical WP8 smoke metrics in `results/WP8/fgsm_robustness_metrics.json`,
* Grad-CAM comparison metadata in `results/gradcam/gradcam_comparison_metadata.json`.

## Remaining TODO

Add confidence intervals or repeated-seed statistics only after such
experiments have actually been run.
