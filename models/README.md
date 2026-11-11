# Trained Model Artifacts (`models/`)

This directory stores serialized PyTorch deep learning weights and model checkpoints for the California wildfire prediction pipeline.

> **Note on File Storage:** Model checkpoints (`*.pt`) contain trained parameter state dictionaries. When cloning the repository or running inference across different environments, ensure that dependencies match (`torch >= 2.2.0` with CUDA or Apple Silicon MPS support).

---

## Active Checkpoint Inventory

| Checkpoint File | Architecture | Training Partition | Seed Configuration | Format | Peak 2023 Val PR-AUC | Final 2023 Val ROC-AUC | Description |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `cnn_baseline_run2_seeded.pt` | `WildfireCNN` | 2022 ($N = 557,616$) | `seed=42` | PyTorch `state_dict` | **0.9214** | **0.9988** | **Canonical baseline:** Fully reproducible run with locked random seeds, dynamic LR decay, and validation loss tracking. |
| `cnn_baseline_run1_unseeded.pt` | `WildfireCNN` | 2022 ($N = 557,616$) | *Unseeded* (System Default) | PyTorch `state_dict` | **0.9317** | **0.9989** | **Proof-of-concept:** Initial exploration run verifying pipeline convergence across 4D spatial tensors. |

---

## Architecture Specification (`WildfireCNN`)

Both checkpoints contain the learned weights for the modular 2D Convolutional Neural Network defined in `src/models.py`:

* **Input Patch Shape:** `(B, 9, 7, 7)` representing a $35\text{ km} \times 35\text{ km}$ spatial bounding patch sampled across 9 standardized physical channels:
  * 4 Weather: 2m Temperature (`t2m`), 2m Relative Humidity (`rh2m`), 10m Wind Speed (`wind_speed_10m`), Total Precipitation (`tp`)
  * 2 Vegetation: `ndvi`, `evi`
  * 3 Topography: `elevation`, `slope`, `aspect`
* **Network Layers:**
  1. `Conv2D(9, 32, kernel_size=3)` $\to$ `BatchNorm2D(32)` $\to$ `ReLU()`
  2. `Conv2D(32, 64, kernel_size=3)` $\to$ `BatchNorm2D(64)` $\to$ `ReLU()`
  3. `AdaptiveAvgPool2D((1, 1))` $\to$ `Flatten()`
  4. `Linear(64, 32)` $\to$ `ReLU()` $\to$ `Dropout(p=0.3)` $\to$ `Linear(32, 1)`

---

## Checkpoint Loading & Inference Snippet

To load either checkpoint for validation diagnostics, feature map visualization, or inference on the 2024 holdout test set (`data/processed/tensors/X_test.npy`):

```python
import torch
from pathlib import Path
from src.models import WildfireCNN

# Select active hardware accelerator
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 1. Instantiate the architecture
model = WildfireCNN().to(device)

# 2. Select checkpoint path
checkpoint_path = Path("../models/cnn_baseline_run2_seeded.pt")

# 3. Load state dictionary
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

# 4. Set to evaluation mode (disables dropout, fixes batchnorm stats)
model.eval()

print(f"Successfully loaded {checkpoint_path.name} onto device: {device}")
```

---

## Associated Diagnostic Artifacts

Numerical loss logs, PR/ROC curves, and validation summary files for these checkpoints are located in `outputs/`:

* `outputs/cnn_baseline_run2_seeded_validation_metrics.json`
* `outputs/cnn_baseline_run2_seeded_diagnostics.png`
* `outputs/cnn_baseline_run1_unseeded_validation_metrics.json`
* `outputs/cnn_baseline_run1_unseeded_diagnostics.png`