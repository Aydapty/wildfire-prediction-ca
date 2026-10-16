# Core Source Code (`src/`)

Modular, reusable Python packages and class definitions powering the California wildfire prediction pipeline.

## Modules

* **`grid_builder.py`**: Contains the `SpatialGrid` class responsible for initializing the fixed 5 km spatial anchor, managing cell indexing (`cell_id`), and performing spatial lookups via `scipy.spatial.KDTree` to map external datasets to nearest grid coordinates.

## Architecture Guidelines

* Modules in `src/` contain class definitions, data structures, and core pipeline logic designed for import across scripts, notebooks, and model training utilities.
* Standalone execution scripts or one-off data fetching scripts should live in `scripts/`, referencing `src/` modules via path imports.

## Usage

Import modules directly into scripts or notebooks