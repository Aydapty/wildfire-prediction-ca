# Execution Drivers (`scripts/`)

Standalone operational scripts for data extraction, pipeline processing, and task execution.

## Available Scripts

* **`download_usgs_dem.py`**: Downloads raw continuous DEM elevation raster data across California from AWS S3 public buckets.
* **`download_era5_api.py`**: Fetches historical meteorological and surface weather grids via CDS API.
* **`build_base_grid.py`**: Instantiates `src.grid_builder.SpatialGrid` to process raw terrain data and generate `data/processed/master_grid_5km.csv`.

## Usage

Run scripts directly from the project root:

```bash
python3 scripts/build_base_grid.py