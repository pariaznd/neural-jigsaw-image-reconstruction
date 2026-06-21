"""General utilities."""

from __future__ import annotations

import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def count_trainable_parameters(model) -> int:
    """Count trainable parameters."""
    return int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))


def print_model_stats(model, max_params: int = 6_000_000) -> None:
    """Print total/trainable parameter counts and check the parameter budget."""
    total_params = model.count_params()
    trainable_params = count_trainable_parameters(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if trainable_params >= max_params:
        raise ValueError(
            f"Model has too many trainable parameters: {trainable_params:,} >= {max_params:,}"
        )

    print(f"OK: trainable parameters are below {max_params:,}.")
