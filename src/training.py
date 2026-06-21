from __future__ import annotations

import numpy as np

from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN
from src.optimizers import SGD


def train_step(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    optimizer: SGD,
    images: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Run one forward, backward, and parameter-update step."""
    logits = model.forward(images)
    loss = loss_function.forward(logits, labels)
    grad_logits = loss_function.backward()
    model.backward(grad_logits)
    parameter_gradient_pairs = model.named_parameters_and_gradients()
    optimizer.step(parameter_gradient_pairs)
    return loss
