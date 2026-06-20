# Neural Jigsaw Image Reconstruction with Differentiable Patch Reassembly

This repository contains a cleaned, GitHub-ready version of a deep learning project for reconstructing full 96×96 RGB images from nine scrambled and cropped 28×28 image patches.

The project was completed as part of a Deep Learning course at NTNU and received **30/30**.

## Problem

Each image is split into a 3×3 grid of 32×32 cells. From each cell, only the central 28×28 patch is provided, and the 9 patches are randomly shuffled.

The model must learn to:

1. infer the spatial arrangement of the patches,
2. reconstruct the missing border content,
3. output the full 96×96 RGB image.

## Architecture

The final model, `JigsawV11b`, combines:

- residual CNN patch encoders,
- edge-aware patch boundary encoding,
- Transformer-style attention for patch-to-patch reasoning,
- differentiable soft patch reassembly,
- a U-Net-style decoder with skip connections.

The model was trained from scratch with no pretrained weights and kept below a 6M trainable-parameter limit.

## Results

| Metric | Value |
|---|---:|
| Test MAE | 0.0292 |
| Test Std | 0.0379 |
| Patch-position accuracy | 88.3% |
| Improvement over baseline | 84.9% |
| Trainable parameters | 5.94M |
| Final grade | 30/30 |

## Repository structure

```text
src/
  config.py       # project configuration
  data.py         # patch extraction and tf.data pipeline
  model.py        # JigsawV11b architecture
  train.py        # training utilities
  evaluate.py     # evaluation utilities
  visualize.py    # qualitative plotting helpers
  utils.py        # seed and parameter-count helpers
requirements.txt
README.md
```

## Minimal usage

```python
from src.config import JigsawConfig
from src.model import build_model
from src.utils import print_model_stats

cfg = JigsawConfig()
model = build_model(cfg)
print_model_stats(model)
```

To train the model, load STL-10 images as a NumPy array with shape `(N, 96, 96, 3)` and call:

```python
from src.train import train_model

model, history = train_model(train_images, val_images, cfg)
```

To evaluate a trained checkpoint:

```python
from src.evaluate import load_model_with_weights, evaluate_model

model = load_model_with_weights("best_jigsaw_v11b_lite_deeper.weights.h5", cfg)
metrics = evaluate_model(model, test_images, cfg)
print(metrics)
```

## Notes

This is a portfolio version of the project. It focuses on the model architecture, reconstruction strategy, and evaluation utilities rather than packaging the original course notebook.
