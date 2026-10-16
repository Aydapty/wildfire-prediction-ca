# wildfire-prediction-ca
COSC 4337 Data Science II (Fall 2026) | University of Houston\
Group 24 Project: Machine learning pipeline predicting daily 24-hour wildfire ignition probabilities across California grid cells using multi-modal environmental data.

---

## Repository Structure

* **`data/`**: Manages raw datasets and processed features (see [`data/README.md`](data/README.md) for full schema details).
* **`docs/`**: Project documentation and architecture notes.
* **`notebooks/`**: Exploratory data analysis and model prototyping notebooks.
* **`scripts/`**: Executable data acquisition, processing, and merge scripts (see [`scripts/README.md`](scripts/README.md)).
* **`src/`**: Reusable core source code and spatial utilities (see [`src/README.md`](src/README.md)).

## Quickstart: Loading the Master Dataset

The preprocessed 2022–2024 spatiotemporal training panel is compiled into an optimized Apache Parquet file. Teammates can load it directly in Python for modeling:

```python
import pandas as pd

# Load the consolidated daily panel (~1.38 GB)
df = pd.read_parquet("data/processed/master_training_dataset.parquet")
print(df.info())
```

## Pipeline Reproduction

To regenerate the processed features and master dataset from scratch, run the scripts in the following order:

```bash
# 1. Build spatial grid and extract DEM baseline
python scripts/build_base_grid.py

# 2. Process individual environmental and target layers
python scripts/process_era5.py
python scripts/process_modis.py
python scripts/map_fpa_fod.py

# 3. Consolidate into the final compressed training panel
python scripts/merge_master_dataset.py