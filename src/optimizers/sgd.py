from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class SGD:
    """Minimal stochastic gradient descent optimizer."""

    def __init__(self, learning_rate: float) -> None:
        try:
            learning_rate = float(learning_rate)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "learning_rate must be a positive finite number."
            ) from error

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError(
                "learning_rate must be a positive finite number."
            )

        self.learning_rate = learning_rate

    def step(
        self,
        parameter_gradient_pairs: Iterable[
            tuple[np.ndarray, np.ndarray]
        ],
    ) -> None:
        updates: list[tuple[np.ndarray, np.ndarray]] = []

        for parameter, gradient in parameter_gradient_pairs:
            if parameter.shape != gradient.shape:
                raise ValueError("Parameter and gradient shapes must match.")
            if not np.isfinite(parameter).all():
                raise ValueError("Parameters must contain only finite values.")
            if not np.isfinite(gradient).all():
                raise ValueError("Gradients must contain only finite values.")

            updated_parameter = parameter - self.learning_rate * gradient
            if not np.isfinite(updated_parameter).all():
                raise ValueError(
                    "SGD update produced non-finite parameter values."
                )
            updates.append((parameter, updated_parameter))

        for parameter, updated_parameter in updates:
            parameter[...] = updated_parameter
