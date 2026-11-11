# Data Directory

This directory manages raw input datasets, interim processed feature files, the consolidated tabular master panel, and pre-extracted 4D spatial tensors for deep learning.

> **Note on Storage:** Raw data files and processed outputs (`.parquet`, `.csv`, `.npy`) are excluded from Git tracking via `.gitignore` to prevent repository bloat.

---

## Directory Structure

```text
data/
├── raw/
│   ├── ecmwf_era5_ranalysis/  # Raw NetCDF/GRIB atmospheric variables
│   ├── fpa_fod/               # SQLite database containing USFS FPA-FOD fire records
│   ├── nasa_mod13q1_v061/     # Raw MODIS vegetation index HDF files
│   └── usgs_dem/              # Raw 1 arc-second DEM elevation GeoTIFF rasters
└── processed/
    ├── era5_daily_features_5km.csv
    ├── fpa_fod_mapped_5km.csv
    ├── master_grid_5km.csv
    ├── modis_daily_features_5km.csv
    ├── master_training_dataset.parquet  # Consolidated primary ML tabular panel (~1.13 GB)
    └── tensors/                         # Pre-extracted 4D spatial arrays for CNN modeling
        ├── X_train.npy                  # Shape: (557616, 9, 7, 7) - 2022 Train features
        ├── y_train.npy                  # Shape: (557616,) - 2022 Train binary labels
        ├── X_val.npy                    # Shape: (1388899, 9, 7, 7) - 2023 Val features
        ├── y_val.npy                    # Shape: (1388899,) - 2023 Val binary labels
        ├── X_test.npy                   # Shape: (1392854, 9, 7, 7) - 2024 Test features
        └── y_test.npy                   # Shape: (1392854,) - 2024 Test binary labels
```

---

## Data Dictionary & Feature Specifications

### 1. `master_grid_5km.csv` (Spatial Anchor)
* **Purpose:** Acts as the fixed spatial reference across California, segmenting the state into uniform 5 km grid cells based on USGS DEM elevation data.
* **Columns:**
  * `cell_id`: Unique integer identifier for each 5 km grid cell.
  * `longitude`: Center longitude coordinate (decimal degrees).
  * `latitude`: Center latitude coordinate (decimal degrees).
  * `elevation`: Mean elevation above sea level (meters); influences local temperature lapse rates and vegetation composition.
  * `slope`: Terrain steepness (degrees); fires spread exponentially faster uphill via radiant heat pre-heating.
  * `aspect`: Compass direction the slope faces (degrees, 0–360); south- and west-facing slopes receive higher solar irradiance and feature drier fuels.

---

### 2. `era5_daily_features_5km.csv` (Meteorological Features)
* **Purpose:** Daily atmospheric variables from ECMWF ERA5 reanalysis driving daily fire weather danger.
* **Columns:**
  * `cell_id`: Foreign key linking to the spatial grid cell.
  * `valid_time`: Daily observation date (`YYYY-MM-DD`).
  * `number`: Ensemble identifier (0 for deterministic reanalysis).
  * `t2m`: 2-meter air temperature (Kelvin); elevated temperatures accelerate fuel moisture depletion.
  * `rh2m`: 2-meter relative humidity (%); critical indicator of dead fuel moisture content.
  * `wind_speed_10m`: Wind speed at 10 meters height (m/s); governs rate of fire spread and flame front oxygenation.
  * `tp`: Total daily precipitation (meters); suppresses active fire behavior and replenishes soil/fuel moisture.

---

### 3. `modis_daily_features_5km.csv` (Vegetation & Fuel Moisture)
* **Purpose:** Surface vegetation density and moisture stress tracked via daily-interpolated 16-day MODIS (MOD13Q1) products.
* **Columns:**
  * `cell_id`: Foreign key linking to the spatial grid cell.
  * `valid_time`: Daily observation date (`YYYY-MM-DD`).
  * `ndvi` (Normalized Difference Vegetation Index): Chlorophyll density metric (-1.0 to 1.0); decreases as live canopy dries into volatile fuel.
  * `evi` (Enhanced Vegetation Index): High-biomass vegetation metric optimized against canopy background and atmospheric aerosol distortions.

---

### 4. `fpa_fod_mapped_5km.csv` (Wildfire Ignition Labels)
* **Purpose:** Point-source wildfire discovery records from the USFS FPA-FOD national database mapped to the nearest 5 km grid cell.
* **Columns:**
  * `FPA_ID`: Unique fire identification string.
  * `FIRE_YEAR`: Calendar year of discovery.
  * `valid_time`: Discovery date normalized to UTC format (`YYYY-MM-DD`).
  * `DISCOVERY_DOY`: Day of year (1–366).
  * `LATITUDE`, `LONGITUDE`: Exact coordinates of fire ignition point.
  * `FIRE_SIZE`: Final burned area in acres.
  * `FIRE_SIZE_CLASS`: Categorical size classification (Class A–G, where A is $< 0.25$ acres and G is $> 5,000$ acres).
  * `NWCG_CAUSE_CLASSIFICATION`: Cause category (Natural vs. Human / Undetermined).
  * `STATE`, `COUNTY`: Administrative regional designations.
  * `cell_id`: Foreign key linking the ignition event to its 5 km grid cell.

---

### 5. `master_training_dataset.parquet` (Consolidated ML Panel)
* **Purpose:** The canonical tabular master dataset combining all spatial, weather, vegetation, and target features across California from 2022 through 2024.
* **Format:** Compressed Apache Parquet format (`zstd` compression, ~1.13 GB; 83,428,616 total cell-day observations across 76,121 active grid cells).
* **Temporal Lag Alignment:** Day $t-1$ environmental conditions are merged with day $t$ wildfire outcomes to ensure predictive models make next-day forecasts without data leakage.
* **Schema:** `cell_id`, `valid_time`, `number`, `t2m`, `rh2m`, `wind_speed_10m`, `tp`, `ndvi`, `evi`, `longitude`, `latitude`, `elevation`, `slope`, `aspect`, `fire_occurred`.

---

### 6. `tensors/` (4D Spatial Deep Learning Tensors)
* **Purpose:** High-dimensional spatial-neighborhood arrays formatted specifically for 2D Convolutional Neural Networks (`WildfireCNN`).
* **Spatial Framing:** Each sample represents a $7 \times 7$ grid cell patch centered on the target cell, corresponding to a **$35\text{ km} \times 35\text{ km}$ spatial receptive field** at 5 km cell resolution. Boundary edges are handled via reflection padding.
* **Channel Stacking (Shape: `(N, 9, 7, 7)`):**
  1. `t2m` (2m Temperature, day $t-1$)
  2. `rh2m` (2m Relative Humidity, day $t-1$)
  3. `wind_speed_10m` (10m Wind Speed, day $t-1$)
  4. `tp` (Total Precipitation, day $t-1$)
  5. `ndvi` (NDVI Vegetation Index, day $t-1$)
  6. `evi` (EVI Vegetation Index, day $t-1$)
  7. `elevation` (USGS DEM Elevation)
  8. `slope` (Terrain Slope)
  9. `aspect` (Terrain Aspect)
* **Split Partitions & Sampling Design:**
  * **2022 Training (`X_train.npy`, `y_train.npy`):**
    * Feature tensor shape: `(557616, 9, 7, 7)` | Label shape: `(557616,)`
    * Composition: 100% of confirmed fire ignitions + deterministic 2% background negative subsample.
  * **2023 Validation (`X_val.npy`, `y_val.npy`):**
    * Feature tensor shape: `(1388899, 9, 7, 7)` | Label shape: `(1388899,)`
    * Composition: 100% of confirmed fire ignitions + deterministic 5% background negative subsample.
  * **2024 Independent Test (`X_test.npy`, `y_test.npy`):**
    * Feature tensor shape: `(1392854, 9, 7, 7)` | Label shape: `(1392854,)`
    * Composition: 100% of confirmed fire ignitions + deterministic 5% background negative subsample.
* **Storage & Memory Optimization:** Serialized as binary NumPy arrays (`.npy`) loaded dynamically via `np.load(..., mmap_mode="r")` inside `WildfireTensorDataset` (`src/dataset.py`).

---

## Acquiring Raw Datasets

To rebuild or update the processed datasets and spatial tensors from scratch, raw data files must be acquired and placed into their respective `data/raw/` subdirectories:

1. **USGS DEM Elevation (`data/raw/usgs_dem/`):**
   * Download 1 arc-second DEM GeoTIFF rasters covering the California bounding box from [USGS The National Map (3DEP)](https://apps.nationalmap.gov/downloader/).
2. **ECMWF ERA5 Reanalysis (`data/raw/ecmwf_era5_ranalysis/`):**
   * Retrieve daily atmospheric variables (2m temperature, 2m dewpoint/relative humidity, 10m wind components, total precipitation) via the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/) API using `scripts/download_era5_api.py`.
3. **NASA MODIS Vegetation (`data/raw/nasa_mod13q1_v061/`):**
   * Download MOD13Q1 v061 250m 16-day NDVI/EVI HDF tiles from [NASA Earthdata / LP DAAC](https://search.earthdata.nasa.gov/).
4. **USFS Fire Program Analysis-FOD (`data/raw/fpa_fod/`):**
   * Download the FPA-FOD SQLite database (`FPA_FOD_20221014.sqlite` or equivalent) from the [USFS Research Data Archive](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6).