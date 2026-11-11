# Execution Scripts (`scripts/`)

This directory contains standalone execution scripts that handle data acquisition, spatial/temporal feature extraction, pipeline consolidation, data quality diagnostics, and 4D spatial tensor extraction for deep learning.

---

## Pipeline Execution Order

Execute the scripts in the following order to build, verify, and transform the master dataset:

### 1. Grid & Baseline Setup
* `build_base_grid.py`: Generates the baseline 5km spatial grid (`master_grid_5km.csv`) covering California using USGS DEM elevation data and establishes reference spatial coordinates (`cell_id`, `latitude`, `longitude`).

### 2. Data Acquisition
* `download_era5_api.py`: Fetches raw ERA5 atmospheric reanalysis variables from the ECMWF CDS API across surface temperature, relative humidity, wind vectors, and total precipitation.
* `download_usgs_dem.py`: Downloads raw 1 arc-second Digital Elevation Model (DEM) tiles from the USGS 3D Elevation Program (3DEP).

### 3. Feature Processing & Mapping
* `process_era5.py`: Processes raw ERA5 atmospheric NetCDF files using direct vectorized `xarray.interp(method="nearest")` to sample daily 5km features across all master grid cells with 100% spatial coverage and built-in spatiotemporal deduplication.
* `process_modis.py`: Aggregates 16-day MODIS vegetation/land product grids (NDVI and EVI) into daily 5km interpolations.
* `map_fpa_fod.py`: Queries the FPA-FOD SQLite database, normalizes Julian ignition dates to standard UTC `valid_time`, filters for the 2022–2024 observation window, and maps discrete fire ignition events to the nearest 5km grid cell.

### 4. Master Consolidation
* `merge_master_dataset.py`: Memory-optimized merge pipeline featuring automated ERA5 and FPA-FOD `groupby` deduplication safety nets, spatiotemporal alignment across day $t-1$ weather and day $t$ wildfire targets, numeric type downcasting, and direct Apache Parquet export (`data/processed/master_training_dataset.parquet`).

### 5. Data Quality & Diagnostics
* `duplicates_culprit_check.py`: Audits individual processed component files independently (`era5`, `modis`, `fpa_fod`, `master_grid`) to isolate source-level spatiotemporal key collisions.
* `duplicates_check.py`: Validates the final master dataset (`master_training_dataset.parquet`) to guarantee zero duplicate `(cell_id, valid_time)` keys and assert continuous data integrity.

### 6. Deep Learning Spatial Tensor Extraction
* `transform_master_dataset.py` *(or `extract_spatial_tensors.py`)*: Extracts 4D spatial-neighborhood tensors from `master_training_dataset.parquet` to serve as inputs for convolutional neural networks (`WildfireCNN`).
  * **Patch Dimensions:** $7 \times 7$ grid patches centered on the target cell ($35\text{ km} \times 35\text{ km}$ spatial receptive field).
  * **Channel Ordering (9 channels):**
    1. `t2m` (2m Temperature)
    2. `rh2m` (2m Relative Humidity)
    3. `wind_speed_10m` (10m Wind Speed)
    4. `tp` (Total Precipitation)
    5. `ndvi` (Normalized Difference Vegetation Index)
    6. `evi` (Enhanced Vegetation Index)
    7. `elevation` (USGS DEM Elevation)
    8. `slope` (Terrain Slope)
    9. `aspect` (Terrain Aspect)
  * **Temporal Alignment:** Features sampled at day $t-1$ paired with binary ignition targets at day $t$.
  * **Output Partitions (`data/processed/tensors/`):**
    * `X_train.npy` `(557616, 9, 7, 7)`, `y_train.npy` `(557616,)` — 2022 (100% fires, 2% background negatives)
    * `X_val.npy` `(1388899, 9, 7, 7)`, `y_val.npy` `(1388899,)` — 2023 (100% fires, 5% background negatives)
    * `X_test.npy` `(1392854, 9, 7, 7)`, `y_test.npy` `(1392854,)` — 2024 (100% fires, 5% background negatives)
  * **Documentation:** Detailed schema and tensor extraction specification located at `docs/transform_master_dataset_breakdown.md`.

---

## Pipeline Summary Table

| Stage | Script | Input Data Source | Output Artifact | Format |
| :--- | :--- | :--- | :--- | :--- |
| **Grid Setup** | `build_base_grid.py` | USGS DEM bounding box | `data/processed/master_grid_5km.csv` | CSV |
| **Acquisition** | `download_era5_api.py` | ECMWF CDS API | `data/raw/era5/` | NetCDF (`.nc`) |
| **Acquisition** | `download_usgs_dem.py` | USGS 3DEP | `data/raw/dem/` | GeoTIFF (`.tif`) |
| **Processing** | `process_era5.py` | `data/raw/era5/`, `master_grid_5km.csv` | `data/interim/era5_processed.parquet` | Parquet |
| **Processing** | `process_modis.py` | MODIS HDF files | `data/interim/modis_processed.parquet` | Parquet |
| **Processing** | `map_fpa_fod.py` | FPA-FOD SQLite DB | `data/interim/fpa_fod_mapped.parquet` | Parquet |
| **Consolidation** | `merge_master_dataset.py` | `data/interim/*` | `data/processed/master_training_dataset.parquet` | Parquet |
| **Diagnostics** | `duplicates_check.py` | `master_training_dataset.parquet` | Integrity pass/fail summary | CLI Log |
| **Tensor Gen** | `transform_master_dataset.py` | `master_training_dataset.parquet` | `data/processed/tensors/*.npy` (6 tensor splits) | NumPy (`.npy`) |