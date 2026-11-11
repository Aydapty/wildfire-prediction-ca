# California Wildfire Prediction (`wildfire-prediction-ca`)

**COSC 4337 Data Science II (Fall 2026) | University of Houston**  
*Group 24 Project:* An end-to-end machine learning and deep learning pipeline predicting next-day (24-hour) wildfire ignition risk across California grid cells using multi-modal environmental, meteorological, and topographic data.

---

## Repository Structure

- **`data/`**: Manages raw environmental inputs, processed tabular panels (`master_training_dataset.parquet`), and pre-extracted 4D spatial tensors (`data/processed/tensors/`) (see [`data/README.md`](data/README.md)).
- **`docs/`**: Academic project proposal, slide deck deliverables, and technical transformation specifications (see [`docs/README.md`](docs/README.md)).
- **`models/`**: Serialized PyTorch model weights (`.pt`) and Scikit-Learn/XGBoost pipelines (see [`models/README.md`](models/README.md)).
- **`notebooks/`**: Interactive experimentation notebooks for classical models, feature ablation, holdout testing, and spatial CNN training (see [`notebooks/README.md`](notebooks/README.md)).
- **`outputs/`**: Central project directory storing validation metrics JSONs, 2024 test evaluations, threshold analysis tables, and diagnostic plots.
- **`scripts/`**: Executable data acquisition, spatial processing, dataset merging, and tensor extraction scripts (see [`scripts/README.md`](scripts/README.md)).
- **`src/`**: Reusable core Python package containing `WildfireCNN`, memory-mapped PyTorch dataset loaders, and spatial utilities (see [`src/README.md`](src/README.md)).

---

## Quickstart: Loading Datasets

### 1. Tabular Master Dataset (Classical ML)
The consolidated 2022–2024 spatiotemporal panel (~1.13 GB compressed Parquet; 83.4M cell-days across 76,121 active 5km grid cells) can be loaded directly:

```python
import pandas as pd

df = pd.read_parquet("data/processed/master_training_dataset.parquet")
print(f"Loaded {len(df):,} cell-day observations across {df['cell_id'].nunique():,} cells.")
```

### 2. 4D Spatial Tensors (Deep Learning)
For spatial modeling, 4D multi-modal tensors ($35\text{ km} \times 35\text{ km}$ patches across 9 physical channels) are streamed via memory-mapping:

```python
from src.dataset import WildfireTensorDataset
from torch.utils.data import DataLoader

train_dataset = WildfireTensorDataset(
    "data/processed/tensors/X_train.npy",
    "data/processed/tensors/y_train.npy"
)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
print(f"Loaded {len(train_dataset):,} spatial training samples.")
```

---

## Data Pipeline & Reproduction

To rebuild the processed features, master dataset, and 4D tensors from scratch, execute the pipeline scripts in order (refer to [`data/README.md`](data/README.md) for raw data acquisition):

```bash
# 1. Build California 5km spatial grid baseline
python scripts/build_base_grid.py

# 2. Process meteorological, vegetation, and historical fire datasets
python scripts/process_era5.py
python scripts/process_modis.py
python scripts/map_fpa_fod.py

# 3. Consolidate into the primary training panel (with deduplication safety nets)
python scripts/merge_master_dataset.py

# 4. Verify data integrity and zero key collisions
python scripts/duplicates_check.py

# 5. Extract 4D spatial tensors (7x7 patches, 9 channels) for CNN training
python scripts/transform_master_dataset.py
```

---

## Model Evaluation and Benchmarks

### Chronological Data Splitting

To prevent temporal leakage and simulate real-world operational forecasting, the project strictly evaluates models across time:

| Split Period | Calendar Year | Target Prevalence | Primary Usage |
| :--- | :---: | :---: | :--- |
| **Training** | 2022 | ~0.65% (subsampled) | Model fitting & feature learning |
| **Validation** | 2023 | ~0.013% (full population) | Hyperparameter tuning, early stopping, threshold calibration |
| **Independent Testing** | 2024 | ~0.014% (full population) | Final out-of-sample holdout benchmarking |

---

### Project Benchmark Comparison

Wildfire ignition is an extreme class imbalance problem ($\sim 0.013\%$ fire prevalence). Because conventional accuracy is misleading on highly skewed distributions, **PR-AUC (Precision-Recall Area Under the Curve)** serves as the primary metric alongside ROC-AUC, Precision, Recall, and F1.

| Architecture / Model Family | Features / Receptive Field | Decision Threshold | 2023 Val PR-AUC | 2023 Val ROC-AUC | 2024 Test PR-AUC | 2024 Test ROC-AUC | 2024 Test F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 9 tabular features (single cell) | 0.0020 | 0.0105 | 0.9646 | 0.0120 | 0.9657 | 0.0220 |
| **XGBoost** | 9 tabular features (single cell) | 0.6900 | 0.2673 | 0.9482 | 0.2704 | 0.9458 | 0.4172 |
| **Random Forest** | 9 tabular features (single cell) | 0.1819 | 0.4493 | 0.9922 | **0.4458** | **0.9910** | **0.4733** |
| **WildfireCNN (Run 2 — Seed 42)** | 9 channels $\times 7 \times 7$ ($35\text{ km}$ spatial patch) | *Optimal AP* | **0.9214** | **0.9988** | *Pending* | *Pending* | *Pending* |

*Note: Initial unseeded CNN baseline (Run 1) verified convergence at **0.9317** Val PR-AUC before seed locking (`seed=42`).*

---

### Key Modeling Insights

1. **Spatial Context is Decisive:** The spatial CNN baseline (`WildfireCNN`) achieves a dramatic lift over all single-pixel tabular baselines (**0.9214** vs. 0.4493 PR-AUC). Capturing moisture gradients, regional fuel connectivity, and surrounding slope orientation across a $35\text{ km} \times 35\text{ km}$ spatial context significantly reduces false positives.
2. **Topography and Vegetation Dominate Tabular Models:** Feature ablation reveals that atmospheric weather variables alone yield near-zero ignition localization (2024 PR-AUC = `0.0006`). Incorporating vegetation (`ndvi`, `evi`) boosts performance to `0.0103`, while static terrain (`elevation`, `slope`, `aspect`) drives the Random Forest to `0.4458`.
3. **Linear Baselines Generate Heavy False Alarms:** Logistic Regression detects 64–66% of fire events but suffers from extreme false-alarm rates (precision $\approx 1.1\%$), reinforcing the necessity of non-linear and deep spatial architectures.

---

## Environment & Dependencies

This repository is verified on **Python 3.12.5 / 3.13.2** across both Apple Silicon (MPS acceleration) and Linux/CUDA environments:

```bash
pip install torch numpy pandas duckdb pyarrow scikit-learn xgboost matplotlib seaborn fastparquet joblib tqdm
```