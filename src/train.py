"""Training utilities for JigsawV11b."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import JigsawConfig
from .data import make_dataset
from .model import build_model


class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""

    def __init__(self, peak: float, warmup_steps: int, total_steps: int, min_lr: float = 5e-7):
        super().__init__()
        self.peak = float(peak)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        warmup_lr = step / self.warmup_steps * self.peak
        cosine_lr = self.peak * 0.5 * (
            1.0 + tf.cos(np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        )
        cosine_lr = tf.maximum(cosine_lr, self.min_lr)

        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak": self.peak,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


def train_model(train_images, val_images, cfg: JigsawConfig = JigsawConfig()):
    """Train JigsawV11b from scratch.

    Args:
        train_images: uint8 numpy array with shape (N, 96, 96, 3)
        val_images: uint8 numpy array with shape (M, 96, 96, 3)
        cfg: project configuration

    Returns:
        model, history
    """
    train_ds = make_dataset(train_images, cfg=cfg, training=True)
    val_ds = make_dataset(val_images, cfg=cfg, training=False)

    model = build_model(cfg)

    steps_per_epoch = int(math.ceil(len(train_images) / cfg.batch_size))
    lr_schedule = WarmupCosine(
        peak=cfg.learning_rate_peak,
        warmup_steps=steps_per_epoch,
        total_steps=cfg.epochs * steps_per_epoch,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=cfg.gradient_clip_norm,
        )
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            cfg.best_weights_path,
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mae",
            mode="min",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(cfg.train_log_path),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


if __name__ == "__main__":
    print(
        "This module provides training utilities. "
        "Load STL-10 images first, then call train_model(train_images, val_images)."
    )
