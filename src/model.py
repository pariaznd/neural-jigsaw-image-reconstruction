"""JigsawV11b model for neural jigsaw image reconstruction.

Architecture summary:
- shared residual CNN patch encoder
- edge encoder for patch boundary cues
- Transformer-style patch reasoning
- differentiable soft patch reassembly
- U-Net-style decoder for full-image reconstruction and inpainting
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import JigsawConfig


def conv_bn_act(x: tf.Tensor, filters: int, kernel_size: int = 3, stride: int = 1) -> tf.Tensor:
    """Conv2D -> BatchNorm -> Swish."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("swish")(x)


def residual_block(x: tf.Tensor, filters: int, stride: int = 1) -> tf.Tensor:
    """Small residual block used in the patch encoder."""
    shortcut = x

    x = conv_bn_act(x, filters, 3, stride)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    return layers.Activation("swish")(layers.Add()([x, shortcut]))


def build_patch_encoder(cfg: JigsawConfig) -> keras.Model:
    """Shared CNN encoder: 28x28 RGB patch -> embedding vector."""
    inputs = layers.Input(shape=(cfg.crop_size, cfg.crop_size, cfg.channels))

    x = conv_bn_act(inputs, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 64, stride=2)    # 14x14
    x = residual_block(x, 96, stride=2)    # 7x7
    x = residual_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(cfg.embed_dim)(x)
    x = layers.LayerNormalization()(x)

    return keras.Model(inputs, x, name="patch_encoder")


def make_mask96(cfg: JigsawConfig) -> tf.Tensor:
    """Static mask: 1 inside cropped patch areas, 0 in missing border regions."""
    mask = np.zeros((cfg.image_size, cfg.image_size, 1), dtype=np.float32)
    for row in range(cfg.grid_size):
        for col in range(cfg.grid_size):
            y0 = row * cfg.patch_size + cfg.margin
            x0 = col * cfg.patch_size + cfg.margin
            mask[y0:y0 + cfg.crop_size, x0:x0 + cfg.crop_size] = 1.0
    return tf.constant(mask[None], dtype=tf.float32)


def soft_canvas_fn(slot_patches: tf.Tensor, cfg: JigsawConfig) -> tf.Tensor:
    """Place 9 ordered patch slots into a 96x96 canvas with zero-padded gaps.

    Args:
        slot_patches: Tensor of shape (B, 9, 28, 28, 3)

    Returns:
        Tensor of shape (B, 96, 96, 3)
    """
    batch_size = tf.shape(slot_patches)[0]
    x = tf.reshape(
        slot_patches,
        (
            batch_size,
            cfg.grid_size,
            cfg.grid_size,
            cfg.crop_size,
            cfg.crop_size,
            cfg.channels,
        ),
    )
    x = tf.pad(
        x,
        [[0, 0], [0, 0], [0, 0], [cfg.margin, cfg.margin], [cfg.margin, cfg.margin], [0, 0]],
        constant_values=0.0,
    )
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, (batch_size, cfg.image_size, cfg.image_size, cfg.channels))


class EdgeEncoder(layers.Layer):
    """Encode thin boundary strips from the four sides of each patch."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(32, 3, padding="same", activation="swish")
        self.conv2 = layers.Conv2D(64, 3, padding="same", activation="swish")
        self.pool = layers.GlobalAveragePooling2D()

    def call(self, patches: tf.Tensor, training: bool = False) -> tf.Tensor:
        edge = 4
        batch_size = tf.shape(patches)[0]

        top = patches[:, :, :edge, :, :]
        bottom = patches[:, :, -edge:, :, :]
        left = tf.transpose(patches[:, :, :, :edge, :], [0, 1, 3, 2, 4])
        right = tf.transpose(patches[:, :, :, -edge:, :], [0, 1, 3, 2, 4])

        edges = tf.concat([top, bottom, left, right], axis=3)
        edges = tf.reshape(edges, (batch_size * 9, 4, 112, 3))

        x = self.conv1(edges)
        x = self.conv2(x)
        x = self.pool(x)

        return tf.reshape(x, (batch_size, 9, 64))


class TransformerBlock(layers.Layer):
    """Pre-norm transformer block for patch-token reasoning."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout,
        )
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(dim),
        ])

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.norm1(x + self.attn(x, x, training=training))
        return self.norm2(x + self.ffn(x, training=training))


class JigsawV11b(keras.Model):
    """Neural jigsaw image reconstruction model.

    The model predicts a soft 9x9 assignment matrix, forms a differentiable
    soft canvas, and decodes the final 96x96 image.
    """

    def __init__(self, cfg: JigsawConfig = JigsawConfig(), **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.mask96 = make_mask96(cfg)

        self.patch_encoder = build_patch_encoder(cfg)
        self.edge_encoder = EdgeEncoder(name="edge_encoder")
        self.token_projection = layers.Dense(cfg.embed_dim, activation="swish")

        self.transformer_blocks = [
            TransformerBlock(
                cfg.embed_dim,
                cfg.num_heads,
                cfg.ff_dim,
                name=f"transformer_block_{i}",
            )
            for i in range(4)
        ]

        # Auxiliary position heads
        self.position_head = keras.Sequential([
            layers.Dense(192, activation="swish"),
            layers.LayerNormalization(),
            layers.Dense(96, activation="swish"),
            layers.Dense(9),
        ], name="position_head")
        self.row_head = keras.Sequential([
            layers.Dense(64, activation="swish"),
            layers.Dense(3),
        ], name="row_head")
        self.col_head = keras.Sequential([
            layers.Dense(64, activation="swish"),
            layers.Dense(3),
        ], name="col_head")

        self.global_context_projection = layers.Dense(32, activation="swish")

        # U-Net-style image-side encoder
        self.e1a = layers.Conv2D(64, 3, padding="same", activation="swish")
        self.e1b = layers.Conv2D(64, 3, padding="same", activation="swish")
        self.e2a = layers.Conv2D(128, 3, strides=2, padding="same", activation="swish")
        self.e2b = layers.Conv2D(128, 3, padding="same", activation="swish")
        self.e3a = layers.Conv2D(192, 3, strides=2, padding="same", activation="swish")
        self.e3b = layers.Conv2D(192, 3, padding="same", activation="swish")
        self.e4a = layers.Conv2D(256, 3, strides=2, padding="same", activation="swish")
        self.e4b = layers.Conv2D(256, 3, padding="same", activation="swish")

        self.bottleneck = layers.Conv2D(256, 3, padding="same", activation="swish")

        # U-Net-style decoder
        self.d4_up = layers.UpSampling2D(2, interpolation="bilinear")
        self.d4a = layers.Conv2D(192, 3, padding="same", activation="swish")
        self.d4b = layers.Conv2D(160, 3, padding="same", activation="swish")
        self.d3_up = layers.UpSampling2D(2, interpolation="bilinear")
        self.d3a = layers.Conv2D(160, 3, padding="same", activation="swish")
        self.d3b = layers.Conv2D(128, 3, padding="same", activation="swish")
        self.d2_up = layers.UpSampling2D(2, interpolation="bilinear")
        self.d2a = layers.Conv2D(96, 3, padding="same", activation="swish")
        self.d2b = layers.Conv2D(64, 3, padding="same", activation="swish")

        self.head1 = layers.Conv2D(32, 3, padding="same", activation="swish")
        self.head2 = layers.Conv2D(3, 1, activation="sigmoid", dtype="float32")

        for name in [
            "loss",
            "mae",
            "position_acc",
            "row_acc",
            "col_acc",
            "ssim_loss",
            "balance_loss",
        ]:
            setattr(self, f"{name}_tracker", keras.metrics.Mean(name=name))

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.mae_tracker,
            self.position_acc_tracker,
            self.row_acc_tracker,
            self.col_acc_tracker,
            self.ssim_loss_tracker,
            self.balance_loss_tracker,
        ]

    def call(self, patches: tf.Tensor, training: bool = False, return_aux: bool = False):
        patches = tf.cast(patches, tf.float32)
        batch_size = tf.shape(patches)[0]

        tokens = self.patch_encoder(
            tf.reshape(patches, (-1, self.cfg.crop_size, self.cfg.crop_size, self.cfg.channels)),
            training=training,
        )
        tokens = tf.reshape(tokens, (batch_size, 9, self.cfg.embed_dim))

        edge_features = self.edge_encoder(patches, training=training)
        x = self.token_projection(tf.concat([tokens, edge_features], axis=-1))

        for block in self.transformer_blocks:
            x = block(x, training=training)

        position_logits = self.position_head(x)
        row_logits = self.row_head(x)
        col_logits = self.col_head(x)

        soft_assignment = tf.nn.softmax(position_logits / self.cfg.temperature, axis=-1)
        soft_slots = tf.einsum("bsp,bshwc->bphwc", soft_assignment, patches)
        soft_canvas = soft_canvas_fn(soft_slots, self.cfg)

        mask = tf.tile(self.mask96, [batch_size, 1, 1, 1])
        decoder_input = tf.concat([soft_canvas, mask], axis=-1)

        s1 = self.e1a(decoder_input)
        s1 = self.e1b(s1)

        s2 = self.e2a(s1)
        s2 = self.e2b(s2)

        s3 = self.e3a(s2)
        s3 = self.e3b(s3)

        s4 = self.e4a(s3)
        s4 = self.e4b(s4)

        z = self.bottleneck(s4)

        global_context = self.global_context_projection(tf.reduce_mean(x, axis=1))
        global_context = tf.reshape(global_context, (batch_size, 1, 1, 32))
        global_context = tf.tile(global_context, [1, tf.shape(z)[1], tf.shape(z)[2], 1])
        z = tf.concat([z, global_context], axis=-1)

        z = self.d4_up(z)
        z = self.d4a(tf.concat([z, s3], axis=-1))
        z = self.d4b(z)

        z = self.d3_up(z)
        z = self.d3a(tf.concat([z, s2], axis=-1))
        z = self.d3b(z)

        z = self.d2_up(z)
        z = self.d2a(tf.concat([z, s1], axis=-1))
        z = self.d2b(z)

        reconstruction = self.head2(self.head1(z))

        if return_aux:
            return reconstruction, position_logits, row_logits, col_logits, soft_assignment, soft_canvas
        return reconstruction

    def compute_losses(
        self,
        patches: tf.Tensor,
        target: tf.Tensor,
        position_labels: tf.Tensor,
        row_labels: tf.Tensor,
        col_labels: tf.Tensor,
        training: bool = False,
    ):
        reconstruction, position_logits, row_logits, col_logits, soft_assignment, _ = self(
            patches,
            training=training,
            return_aux=True,
        )

        target = tf.cast(target, tf.float32)
        reconstruction = tf.cast(reconstruction, tf.float32)

        mae = tf.reduce_mean(tf.abs(target - reconstruction))
        ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(target, reconstruction, max_val=1.0))

        position_loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(
                position_labels,
                position_logits,
                from_logits=True,
            )
        )
        row_loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(
                row_labels,
                row_logits,
                from_logits=True,
            )
        )
        col_loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(
                col_labels,
                col_logits,
                from_logits=True,
            )
        )

        col_sums = tf.reduce_sum(soft_assignment, axis=1)
        balance_loss = tf.reduce_mean(tf.square(col_sums - 1.0))

        soft_grid = tf.reshape(soft_assignment, (-1, 9, 3, 3))
        balance_loss += 0.3 * tf.reduce_mean(tf.square(tf.reduce_sum(soft_grid, axis=[1, 3]) - 3.0))
        balance_loss += 0.3 * tf.reduce_mean(tf.square(tf.reduce_sum(soft_grid, axis=[1, 2]) - 3.0))

        total = (
            self.cfg.w_mae * mae
            + self.cfg.w_ssim * ssim_loss
            + self.cfg.w_pos * position_loss
            + self.cfg.w_row * row_loss
            + self.cfg.w_col * col_loss
            + self.cfg.w_balance * balance_loss
        )

        position_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(position_logits, axis=-1, output_type=tf.int32),
                    position_labels,
                ),
                tf.float32,
            )
        )
        row_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(row_logits, axis=-1, output_type=tf.int32), row_labels),
                tf.float32,
            )
        )
        col_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(col_logits, axis=-1, output_type=tf.int32), col_labels),
                tf.float32,
            )
        )

        return total, mae, ssim_loss, balance_loss, position_acc, row_acc, col_acc, reconstruction

    def _update_metrics(self, total, mae, ssim_loss, balance_loss, position_acc, row_acc, col_acc):
        self.loss_tracker.update_state(total)
        self.mae_tracker.update_state(mae)
        self.ssim_loss_tracker.update_state(ssim_loss)
        self.balance_loss_tracker.update_state(balance_loss)
        self.position_acc_tracker.update_state(position_acc)
        self.row_acc_tracker.update_state(row_acc)
        self.col_acc_tracker.update_state(col_acc)

    def train_step(self, data):
        patches, targets = data

        with tf.GradientTape() as tape:
            total, mae, ssim_loss, balance_loss, position_acc, row_acc, col_acc, _ = self.compute_losses(
                patches,
                targets["image"],
                targets["positions"],
                targets["rows"],
                targets["cols"],
                training=True,
            )

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self._update_metrics(total, mae, ssim_loss, balance_loss, position_acc, row_acc, col_acc)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        patches, targets = data

        total, mae, ssim_loss, balance_loss, position_acc, row_acc, col_acc, _ = self.compute_losses(
            patches,
            targets["image"],
            targets["positions"],
            targets["rows"],
            targets["cols"],
            training=False,
        )

        self._update_metrics(total, mae, ssim_loss, balance_loss, position_acc, row_acc, col_acc)
        return {metric.name: metric.result() for metric in self.metrics}

    def get_config(self):
        return {
            "cfg": self.cfg.__dict__,
            **super().get_config(),
        }


def build_model(cfg: JigsawConfig = JigsawConfig(), name: str = "jigsaw_v11b") -> JigsawV11b:
    """Build and initialize the model with a dummy input."""
    model = JigsawV11b(cfg=cfg, name=name)
    _ = model(tf.zeros((1, 9, cfg.crop_size, cfg.crop_size, cfg.channels), dtype=tf.float32))
    return model
