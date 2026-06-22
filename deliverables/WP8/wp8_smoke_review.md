# WP8 Controlled Smoke Evaluation Review

## Scope

This note reviews the controlled tiny local FGSM robustness smoke evaluation.
It records pipeline behavior and the decision boundary for any larger
evaluation. It is not the final WP8 summary.

## Controlled Run

The smoke evaluation used:

```text
eval_samples: 32
batch_size: 8
seed: 42
epsilon_values: [0, 2/255, 4/255, 8/255, 16/255]
representative_epsilon: 8/255
```

The run used the existing local checkpoint and local CIFAR-10 data. It did not
download data, train the model, modify a checkpoint, use ZITI, or run a larger
evaluation.

## Artifacts

```text
results/WP8/fgsm_robustness_metrics.json
results/WP8/fgsm_accuracy_vs_epsilon.png
```

## Result Summary

For every evaluated epsilon:

```text
clean_accuracy: 0.0
adversarial_accuracy: 0.0
accuracy_drop: 0.0
attack_success_rate: 0.0
```

The representative-example result was:

```text
successful: []
failed: []
```

## Interpretation

The WP8 evaluation pipeline runs end to end and produces the expected metrics
and plot artifacts.

The current controlled checkpoint is too weak on this fixed 32-sample subset:
`clean_correct_samples` is zero. Consequently, `attack_success_rate` is
defined as `0.0`, and representative-example selection has no eligible
clean-correct samples.

This smoke result validates pipeline execution only. It must not be used as a
meaningful CIFAR-10 robustness conclusion.

## Larger-Evaluation Decision

No larger evaluation was run, and ZITI was not used in this step.

A larger formal evaluation should be considered only after explicit user
confirmation, ideally with a more credible baseline checkpoint. Before any
many-image evaluation, repeated-seed run, or larger epsilon sweep, ask whether
to use the university-provided ZITI cluster.
