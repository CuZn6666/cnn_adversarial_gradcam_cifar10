# From-Scratch CNN Robustness Analysis on CIFAR-10

## Project Objective

This project implements a compact convolutional neural network from scratch
for CIFAR-10 image classification. Based on the trained baseline model, we
analyze adversarial robustness using gradient-based and non-gradient-based
attacks, and qualitatively investigate attention changes using Grad-CAM.

## Selected Methods

### Model
- Compact CNN implemented from scratch
- Dataset: CIFAR-10

### Adversarial Attacks
- FGSM: one-step gradient-based white-box attack
- PGD: iterative gradient-based white-box attack
- Simplified square-based attack: non-gradient black-box attack

### Explainability
- Grad-CAM for qualitative comparison of clean and adversarial examples

## Evaluation Metrics

- Clean accuracy
- Adversarial accuracy
- Accuracy drop
- Attack success rate
- Query count for the black-box attack

## Current Project Phase

- WP0: Focused literature review and final method selection
- WP1: Project setup and CIFAR-10 pipeline

## Next Step

- WP2: Compact CNN forward implementation

## Repository Structure

```text
configs/          Configuration files
src/              Source code
experiments/      Experiment entry scripts
tests/            Unit and sanity tests
results/          Figures, tables, logs and checkpoints
deliverables/     Work-package-specific intermediate outputs