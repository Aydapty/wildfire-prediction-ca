# Execution Scripts

This directory contains standalone execution scripts that handle data acquisition, spatial/temporal feature extraction, and pipeline consolidation.

## Pipeline Order

Run the scripts in the following order to build the master dataset:

1. **Grid & Baseline Setup**
   * `build_base_grid.py`: Generates the baseline 5km spatial grid (`master_grid_5km.csv`) covering California using USGS DEM elevation data.

2. **Data Acquisition**
   * `download_era5_api.py`: Fetches raw ERA5 atmospheric reanalysis variables.
   * `download_usgs_dem.py`: Downloads raw Digital Elevation Model (DEM) tiles.

3. **Feature Processing & Mapping**
   * `process_era5.py`: Processes raw ERA5 atmospheric files into daily 5km aggregated features.
   * `process_modis.py`: Aggregates 16-day MODIS vegetation/land product grids into daily 5km interpolations.
   * `map_fpa_fod.py`: Queries the FPA-FOD SQLite database, normalizes Julian ignition dates to `valid_time`, filters for the 2022–2024 window, and maps fire events to the nearest 5km grid cell.

4. **Master Consolidation**
   * `merge_master_dataset.py`: Memory-optimized merge script that aligns spatiotemporal layers, downcasts numeric types, cleans missing target flags, and outputs the final training dataset directly to Apache Parquet (`master_training_dataset.parquet`).