# Project Scope Draft

## Objective

The objective of this project is to implement a compact CNN from scratch
for CIFAR-10 classification and analyze its vulnerability to adversarial
perturbations.

## Planned Components

### Baseline Model
- Compact CNN implemented from scratch
- Training and evaluation on CIFAR-10

### Gradient-Based Attacks
- FGSM
- PGD

### Non-Gradient-Based Attack
- Simplified square-based black-box attack

### Explainability
- Grad-CAM visualization for clean and adversarial examples

## Evaluation Criteria

- Clean test accuracy
- Adversarial accuracy under different perturbation strengths
- Accuracy drop
- Attack success rate on originally correctly classified samples
- Query count for the black-box attack

## Current Scope Restrictions

- No large-scale adversarial training
- No extensive comparison across deep learning backends
- Grad-CAM is used qualitatively, not as a quantitative evaluation metric
