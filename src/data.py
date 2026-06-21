"""Data pipeline for reconstructing images from scrambled cropped patches.

The model receives 9 scrambled 28x28 RGB patches and reconstructs the original
96x96 RGB image. Each patch is cropped from the center of a 32x32 cell in a 3x3
grid, so the boundary pixels between cells are missing and must be reconstructed.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from .config import JigsawConfig


AUTOTUNE = tf.data.AUTOTUNE


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """Light image augmentation used during training."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.05)
    image = tf.image.random_contrast(image, 0.90, 1.10)
    return tf.clip_by_value(image, 0.0, 1.0)


def extract_patches(image: tf.Tensor, cfg: JigsawConfig = JigsawConfig()) -> tf.Tensor:
    """Extract the 9 ordered cropped patches from a 96x96 image.

    Returns:
        Tensor with shape (9, 28, 28, 3), ordered row-major.
    """
    patches = []
    for row in range(cfg.grid_size):
        for col in range(cfg.grid_size):
            y0 = row * cfg.patch_size + cfg.margin
            x0 = col * cfg.patch_size + cfg.margin
            patch = image[y0:y0 + cfg.crop_size, x0:x0 + cfg.crop_size, :]
            patches.append(patch)
    return tf.stack(patches, axis=0)


def process_train(image: tf.Tensor, cfg: JigsawConfig = JigsawConfig()) -> Tuple[tf.Tensor, dict]:
    """Create one scrambled training sample."""
    image = augment_image(tf.cast(image, tf.float32) / 255.0)
    ordered_patches = extract_patches(image, cfg)
    perm = tf.random.shuffle(tf.range(cfg.grid_size * cfg.grid_size, dtype=tf.int32))

    return tf.gather(ordered_patches, perm), {
        "image": image,
        "positions": perm,
        "rows": perm // cfg.grid_size,
        "cols": perm % cfg.grid_size,
    }


def process_eval(image: tf.Tensor, cfg: JigsawConfig = JigsawConfig()) -> Tuple[tf.Tensor, dict]:
    """Create one scrambled validation/test sample."""
    image = tf.cast(image, tf.float32) / 255.0
    ordered_patches = extract_patches(image, cfg)
    perm = tf.random.shuffle(tf.range(cfg.grid_size * cfg.grid_size, dtype=tf.int32))

    return tf.gather(ordered_patches, perm), {
        "image": image,
        "positions": perm,
        "rows": perm // cfg.grid_size,
        "cols": perm % cfg.grid_size,
    }


def make_dataset(
    images,
    cfg: JigsawConfig = JigsawConfig(),
    training: bool = True,
    shuffle_buffer: int = 6000,
) -> tf.data.Dataset:
    """Build a tf.data pipeline from a numpy array or tensor of images."""
    dataset = tf.data.Dataset.from_tensor_slices(images)

    if training:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    map_fn = process_train if training else process_eval
    dataset = dataset.map(lambda x: map_fn(x, cfg), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(cfg.batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return dataset
