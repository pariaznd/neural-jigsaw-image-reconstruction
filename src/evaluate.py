"""Evaluation utilities for the jigsaw reconstruction model."""

from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf

from .config import JigsawConfig
from .data import make_dataset
from .model import build_model


def evaluate_model(model, test_images, cfg: JigsawConfig = JigsawConfig()) -> Dict[str, float]:
    """Evaluate per-image MAE and auxiliary placement accuracies."""
    test_ds = make_dataset(test_images, cfg=cfg, training=False)

    mae_values = []
    pos_values = []
    row_values = []
    col_values = []

    for patches, targets in test_ds:
        recon, pos_logits, row_logits, col_logits, _, _ = model(
            patches,
            training=False,
            return_aux=True,
        )

        batch_mae = tf.reduce_mean(
            tf.abs(tf.cast(targets["image"], tf.float32) - tf.cast(recon, tf.float32)),
            axis=[1, 2, 3],
        )
        mae_values.extend(batch_mae.numpy().tolist())

        pos_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(pos_logits, axis=-1, output_type=tf.int32),
                    targets["positions"],
                ),
                tf.float32,
            ),
            axis=1,
        )
        row_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(row_logits, axis=-1, output_type=tf.int32), targets["rows"]),
                tf.float32,
            ),
            axis=1,
        )
        col_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(col_logits, axis=-1, output_type=tf.int32), targets["cols"]),
                tf.float32,
            ),
            axis=1,
        )

        pos_values.extend(pos_acc.numpy().tolist())
        row_values.extend(row_acc.numpy().tolist())
        col_values.extend(col_acc.numpy().tolist())

    mae_values = np.asarray(mae_values, dtype=np.float32)

    return {
        "test_mae": float(mae_values.mean()),
        "test_std": float(mae_values.std()),
        "position_accuracy": float(np.mean(pos_values)),
        "row_accuracy": float(np.mean(row_values)),
        "col_accuracy": float(np.mean(col_values)),
    }


def load_model_with_weights(
    weights_path: str,
    cfg: JigsawConfig = JigsawConfig(),
):
    """Build a fresh model and load trained weights."""
    model = build_model(cfg)
    model.load_weights(weights_path)
    return model


if __name__ == "__main__":
    print(
        "This module provides evaluation utilities. "
        "Build/load a model, then call evaluate_model(model, test_images)."
    )
