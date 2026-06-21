"""Visualization helpers for reconstructed jigsaw images."""

from __future__ import annotations

import matplotlib.pyplot as plt
import tensorflow as tf

from .data import make_dataset
from .config import JigsawConfig


def plot_reconstructions(model, images, cfg: JigsawConfig = JigsawConfig(), num_examples: int = 5):
    """Plot scrambled patches, reconstruction, and ground truth."""
    ds = make_dataset(images, cfg=cfg, training=False)

    patches, targets = next(iter(ds))
    reconstruction = model(patches, training=False)

    num_examples = min(num_examples, patches.shape[0])

    for i in range(num_examples):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        # Show the 9 scrambled patches as a compact 3x3 grid.
        patch_grid = tf.reshape(patches[i], (3, 3, cfg.crop_size, cfg.crop_size, cfg.channels))
        patch_grid = tf.transpose(patch_grid, [0, 2, 1, 3, 4])
        patch_grid = tf.reshape(patch_grid, (3 * cfg.crop_size, 3 * cfg.crop_size, cfg.channels))

        axes[0].imshow(patch_grid)
        axes[0].set_title("Scrambled patches")
        axes[0].axis("off")

        axes[1].imshow(reconstruction[i])
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")

        axes[2].imshow(targets["image"][i])
        axes[2].set_title("Ground truth")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
