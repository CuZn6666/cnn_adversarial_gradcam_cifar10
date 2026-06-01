from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seeds used by the current NumPy-based project.

    Args:
        seed: Fixed integer seed for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    