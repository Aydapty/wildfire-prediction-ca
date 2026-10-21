# Data Directory

This directory manages raw input datasets and processed feature outputs. 

> **Note:** Raw data files and processed outputs are excluded from version control via `.gitignore`.

## Structure

```text
data/
├── raw/
│   ├── ecmwf_era5_ranalysis/  # Raw NetCDF/GRIB atmospheric variables
│   ├── fpa_fod/               # SQLite database containing FPA-FOD fire records
│   ├── nasa_mod13q1_v061/     # Raw MODIS vegetation index files
│   └── usgs_dem/              # Raw DEM elevation rasters
└── processed/
    ├── era5_daily_features_5km.csv
    ├── fpa_fod_mapped_5km.csv
    ├── master_grid_5km.csv
    ├── modis_daily_features_5km.csv
    └── master_training_dataset.parquet  # Compressed primary ML training file
```

---

## Data Dictionary & Feature Specifications

### 1. `master_grid_5km.csv` (Spatial Baseline)
* **Purpose:** Acts as the fixed spatial anchor across California, dividing the region into uniform 5km grid cells using USGS DEM data.
* **Columns:**
  * `cell_id`: Unique numeric identifier for each 5km grid cell.
  * `longitude`: Center longitude coordinate (decimal degrees).
  * `latitude`: Center latitude coordinate (decimal degrees).
  * `elevation`: Mean elevation above sea level (meters); influences local climate, vegetation type, and oxygen levels.
  * `slope`: Incline or steepness of the terrain (degrees); fires spread significantly faster uphill due to heat convection.
  * `aspect`: Compass direction the slope faces (degrees); south/west-facing slopes receive more sunlight, resulting in drier fuels.

---

### 2. `era5_daily_features_5km.csv` (Meteorological Data)
* **Purpose:** Captures weather-driven risk factors (temperature, humidity, wind, precipitation) influencing daily fire weather.
* **Columns:**
  * `latitude`, `longitude`: Spatial coordinates matching the ERA5 sample point.
  * `valid_time`: Daily observation date (`YYYY-MM-DD`).
  * `number`: Ensemble member identifier (0 for deterministic reanalysis).
  * `t2m`: 2-meter air temperature (Kelvin/Celsius); higher temperatures accelerate fuel drying and ignitability.
  * `rh2m`: 2-meter relative humidity (%); low humidity drops fuel moisture content, drastically increasing ignition risk.
  * `wind_speed_10m`: Wind speed at 10 meters height (m/s); drives fire spread rate, direction, and oxygen supply.
  * `tp`: Total daily precipitation (meters/mm); suppresses fire activity and increases fuel moisture over time.
  * `cell_id`: Foreign key linking to the spatial grid cell.

---

### 3. `modis_daily_features_5km.csv` (Vegetation & Fuel Data)
* **Purpose:** Tracks surface vegetation health and fuel availability using daily-interpolated MODIS land-product indices.
* **Columns:**
  * `valid_time`: Daily observation date (`YYYY-MM-DD`).
  * `cell_id`: Foreign key linking to the spatial grid cell.
  * `ndvi` (Normalized Difference Vegetation Index): Measures live green vegetation density (-1 to 1); drops during droughts as vegetation dries into combustible fuel.
  * `evi` (Enhanced Vegetation Index): Enhanced vegetation metric optimized for high-biomass regions with less atmospheric distortion.

---

### 4. `fpa_fod_mapped_5km.csv` (Wildfire Target Labels)
* **Purpose:** Historical wildfire ignition records from the FPA-FOD database, normalized and mapped to nearest grid cells.
* **Columns:**
  * `FPA_ID`: Unique national fire record identifier from the USFS FPA-FOD database.
  * `FIRE_YEAR`: Calendar year of wildfire discovery.
  * `valid_time`: Discovery date normalized to daily format (`YYYY-MM-DD`).
  * `DISCOVERY_DOY`: Day of year (1–365/366) of fire discovery.
  * `LATITUDE`, `LONGITUDE`: Exact point coordinates of original fire ignition.
  * `FIRE_SIZE`: Final burned area size in acres.
  * `FIRE_SIZE_CLASS`: Categorical size classification (Class A–G, where A is < 0.25 acres and G is > 5,000 acres).
  * `NWCG_CAUSE_CLASSIFICATION`: General cause category (e.g., Natural vs. Human Or Undetermined).
  * `STATE`, `COUNTY`: Administrative location boundaries.
  * `cell_id`: Foreign key linking the ignition event to its 5km grid cell.

---

### 5. `master_training_dataset.parquet` (Primary Training Panel)
* **Purpose:** Consolidated, memory-optimized spatiotemporal panel (2022–2024) containing all merged spatial, weather, vegetation, and target features.
* **Format:** Apache Parquet binary format (~1.38 GB) for fast loading and low RAM overhead.
* **Spatiotemporal Structure & Relational Design:**
  * **Panel Layout:** The dataset is structured as a spatiotemporal panel where each unique `cell_id` spans multiple rows corresponding to every calendar day (`valid_time`) from 2022 through 2024.
  * **Constant vs. Time-Varying Data:** Static geographical attributes (such as `elevation`, `slope`, `aspect`, and base coordinates) remain identical across all rows for a given `cell_id`. Meanwhile, environmental and target attributes (`t2m`, `rh2m`, `wind_speed_10m`, `tp`, `ndvi`, `evi`, and fire ignition flags) update daily to reflect changing weather conditions, moisture stress, and historical wildfire events.
* **Key Columns & Feature Schema:**
  * `cell_id`: Unique numeric identifier for each 5km grid cell.
  * `valid_time`: Observation date (`YYYY-MM-DD`).
  * `latitude`, `longitude`: Grid cell center coordinates.
  * `elevation`, `slope`, `aspect`: Static terrain features (from USGS DEM).
  * `t2m`, `rh2m`, `wind_speed_10m`, `tp`: Daily meteorological features (from ERA5).
  * `ndvi`, `evi`: Daily vegetation & fuel moisture indices (from MODIS).
  * `fire_occurred`: Target variable — binary indicator (`1` if at least one wildfire ignition occurred in that 5km cell on that day, `0` otherwise; derived from USFS FPA-FOD records aggregated via `.max()`).

---

## Acquiring Raw Datasets

Because raw data files and processed outputs are excluded from version control via `.gitignore`, external users or reviewers must download the source files independently and place them into the corresponding `data/raw/` subdirectories before running the processing scripts. 

> **Target Scope:** Raw datasets must be clipped or filtered to cover the **California** regional bounding box for the **2022–2024** study window to successfully reproduce the training panel.

### 1. USGS DEM Elevation Data (`data/raw/usgs_dem/`)
* **Source:** [USGS EarthExplorer](https://earthexplorer.usgs.gov/) or The National Map.
* **Instructions:** Download digital elevation model (DEM) raster files covering California and place the raw `.tif` files into `data/raw/usgs_dem/`.

### 2. ECMWF ERA5 Reanalysis (`data/raw/ecmwf_era5_ranalysis/`)
* **Source:** [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/)
* **Instructions:** Use the `cdsapi` Python package to download daily meteorological variables for the California bounding box across 2022–2024 into `data/raw/ecmwf_era5_ranalysis/`.

### 3. NASA MODIS Vegetation Indices (`data/raw/nasa_mod13q1_v061/`)
* **Source:** [NASA Land Processes Distributed Active Archive Center (LP DAAC)](https://lpdaac.usgs.gov/) via NASA Earthdata.
* **Instructions:** Download MOD13Q1 v061 (NDVI/EVI raster products) for California and save them in `data/raw/nasa_mod13q1_v061/`.

### 4. USFS Fire Program Analysis-FOD (`data/raw/fpa_fod/`)
* **Source:** [Kaggle (USDA Forest Service Wildfire Dataset)](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires) or the [USFS Data Library](https://www.fs.usda.gov/).
* **Instructions:** Download the SQLite database version (`FPA_FOD_20170508.sqlite` or later revision) and place it directly inside `data/raw/fpa_fod/` (the spatial filter is handled automatically during script execution).