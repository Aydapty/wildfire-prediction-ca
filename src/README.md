# Core Source Code (`src/`)

Modular, reusable Python packages, dataset loaders, neural network architectures, and spatial utilities powering the California wildfire ignition risk prediction pipeline.

---

## Package Overview

The `src/` directory is structured as an installable Python package (`src/__init__.py`) designed for clean, unified imports across both execution scripts in `scripts/` and interactive notebooks in `notebooks/`.

```
src/
├── __init__.py           # Package namespace initialization & direct symbol exports
├── dataset.py            # Memory-mapped PyTorch dataset for 4D multi-modal spatial tensors
├── models.py             # Custom PyTorch deep learning architectures (WildfireCNN)
├── grid_builder.py       # 5 km California anchor grid indexing & KDTree spatial lookups
└── utils.py              # Timestamped pipeline logging and diagnostic utilities
```

---

## Module Reference

### 1. `src/models.py` — Deep Learning Architectures
Houses custom PyTorch neural network architectures engineered for spatial wildfire ignition modeling:

* **`WildfireCNN`**: Modular 2D Convolutional Neural Network designed to process multi-channel spatial neighborhood patches:
  * **Input Tensor Shape:** `(B, 9, 7, 7)` representing 9 physical channels across a $35\text{ km} \times 35\text{ km}$ spatial grid ($7 \times 7$ at 5 km cell resolution).
  * **Feature Extractor:**
    * `Conv2D(in_channels=9, out_channels=32, kernel_size=3, padding=0)` $\to$ `BatchNorm2d(32)` $\to$ `ReLU()` (reduces $7 \times 7 \to 5 \times 5$)
    * `Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=0)` $\to$ `BatchNorm2d(64)` $\to$ `ReLU()` (reduces $5 \times 5 \to 3 \times 3$)
    * `AdaptiveAvgPool2d((1, 1))` pooling spatial feature maps into a 64-dimensional channel summary vector.
  * **Classifier Head:**
    * `Flatten(start_dim=1)`
    * `Linear(in_features=64, out_features=32)` $\to$ `ReLU()`
    * `Dropout(p=0.3)`
    * `Linear(in_features=32, out_features=1)` (outputs raw logits for numerically stable `BCEWithLogitsLoss`).

### 2. `src/dataset.py` — PyTorch Dataset & Streaming
Provides optimized tensor-streaming classes tailored for large-scale geospatial memory partitions:

* **`WildfireTensorDataset`**: Subclasses `torch.utils.data.Dataset` to stream pre-extracted 4D NumPy feature arrays (`X_*.npy`) and binary label arrays (`y_*.npy`):
  * **Memory-Mapped Loading:** Utilizes `np.load(..., mmap_mode="r")` to avoid loading multi-gigabyte partitions into active RAM simultaneously, enabling fast random mini-batch sampling across millions of observations on Apple Silicon GPUs (MPS), NVIDIA CUDA, and CPU.
  * **Format & Typing:** Casts feature tensors to `torch.float32` and labels to `(1,)` shaped `torch.float32` vectors suitable for `nn.BCEWithLogitsLoss(pos_weight=...)`.

### 3. `src/grid_builder.py` — Spatial Anchor & Coordinate Mapping
* **`SpatialGrid`**: Initializes the baseline 5 km California grid anchor (`master_grid_5km.csv`), manages unique cell indexing (`cell_id`), and provides fast coordinate spatial lookups via `scipy.spatial.KDTree` to map raw observational datasets (ERA5, MODIS, FPA-FOD) to the nearest spatial grid point.

### 4. `src/utils.py` — Diagnostic & Logging Utilities
* **`log`**: Modular console logging utility that timestamps standard output with microsecond precision, ensuring reproducible tracking during data preprocessing, merging, and model training.

---

## Import Configurations

### 1. In Execution Scripts (`scripts/`)
Add the parent project directory to `sys.path` to enable clean package-style imports:

```python
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import log
from src.grid_builder import SpatialGrid
from src.models import WildfireCNN
from src.dataset import WildfireTensorDataset
```

### 2. In Jupyter Notebooks (`notebooks/`)
Configure autoreload and path resolution at the top of notebook environments:

```python
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import WildfireCNN
from src.dataset import WildfireTensorDataset
```