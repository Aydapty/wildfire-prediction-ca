# Execution Scripts

This directory contains standalone execution scripts that handle data acquisition, spatial/temporal feature extraction, pipeline consolidation, and data quality diagnostics.

## Pipeline Order

Run the scripts in the following order to build and verify the master dataset:

1. **Grid & Baseline Setup**
   * `build_base_grid.py`: Generates the baseline 5km spatial grid (`master_grid_5km.csv`) covering California using USGS DEM elevation data.

2. **Data Acquisition**
   * `download_era5_api.py`: Fetches raw ERA5 atmospheric reanalysis variables.
   * `download_usgs_dem.py`: Downloads raw Digital Elevation Model (DEM) tiles.

3. **Feature Processing & Mapping**
   * `process_era5.py`: Processes raw ERA5 atmospheric files using direct vectorized `xarray.interp(method="nearest")` to sample daily 5km features across all master grid cells with 100% spatial coverage and built-in spatiotemporal deduplication.
   * `process_modis.py`: Aggregates 16-day MODIS vegetation/land product grids into daily 5km interpolations.
   * `map_fpa_fod.py`: Queries the FPA-FOD SQLite database, normalizes Julian ignition dates to `valid_time`, filters for the 2022–2024 window, and maps fire events to the nearest 5km grid cell.

4. **Master Consolidation**
   * `merge_master_dataset.py`: Memory-optimized merge script featuring automated ERA5 and FPA-FOD groupby deduplication safety nets, spatiotemporal alignment, type downcasting, and direct Apache Parquet export (`master_training_dataset.parquet`).

5. **Data Quality & Diagnostics**
   * `duplicates_culprit_check.py`: Audits individual processed component files independently (`era5`, `modis`, `fpa_fod`, `master_grid`) to isolate source-level spatiotemporal key collisions.
   * `duplicates_check.py`: Validates the final master dataset (`master_training_dataset.parquet`) to guarantee zero duplicate `(cell_id, valid_time)` keys and identical row entries.