"""Configuration for the neural jigsaw image reconstruction project."""

from dataclasses import dataclass


@dataclass
class JigsawConfig:
    # Data geometry
    image_size: int = 96
    grid_size: int = 3
    patch_size: int = 32
    crop_size: int = 28
    margin: int = 2
    channels: int = 3

    # Model
    embed_dim: int = 160
    num_heads: int = 4
    ff_dim: int = 384
    temperature: float = 0.25

    # Training
    batch_size: int = 96
    epochs: int = 32
    learning_rate_peak: float = 6e-4
    gradient_clip_norm: float = 0.7

    # Loss weights
    w_mae: float = 1.00
    w_ssim: float = 0.20
    w_pos: float = 0.20
    w_row: float = 0.05
    w_col: float = 0.05
    w_balance: float = 0.20

    # Reproducibility / artifact names
    seed: int = 42
    best_weights_path: str = "best_jigsaw_v11b_lite_deeper.weights.h5"
    train_log_path: str = "training_log.csv"
