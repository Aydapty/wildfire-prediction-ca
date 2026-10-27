import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.grid_builder import SpatialGrid
from src.utils import log

def main():
    log("Loading master spatial grid anchor...")
    master_grid_path = os.path.join(PROJECT_ROOT, "data", "processed", "master_grid_5km.csv")
    grid = SpatialGrid(master_grid_path)
    modis_dir = Path("data/raw")

    # Find the MODIS NetCDF file recursively across subfolders
    modis_files = list(modis_dir.glob("**/MOD13Q1*.nc"))
    if not modis_files:
        raise FileNotFoundError("No MODIS NetCDF file found in data/raw/")

    modis_path = modis_files[0]
    log(f"Opening MODIS dataset lazily: {modis_path.name}")

    ds = xr.open_dataset(modis_path)
    log("Dataset variables available:", list(ds.variables))

    # Extract unique years and filter strictly for 2022-2024
    time_vals = ds['time'].values
    all_years = sorted(list(set(pd.to_datetime([str(t) for t in time_vals]).year)))
    years = [y for y in all_years if 2022 <= y <= 2024]
    log(f"Detected target years to process: {years}")

    all_processed_frames = []

    for year in years:
        log(f"\n--- Processing Year {year} ---")
        ds_year = ds.sel(time=str(year))

        lats = ds_year['lat'].values
        chunk_size = 50  # Number of latitude rows per micro-chunk

        for i in range(0, len(lats), chunk_size):
            lat_slice = lats[i:i + chunk_size]
            log(f"  -> Processing latitude band {i}/{len(lats)}...")

            ds_band = ds_year.sel(lat=lat_slice)

            # Apply QA mask locally
            if '_250m_16_days_pixel_reliability' in ds_band.variables:
                reliability = ds_band['_250m_16_days_pixel_reliability']
                good_quality = (reliability <= 1)
                ndvi = ds_band['_250m_16_days_NDVI'].where(good_quality)
                evi = ds_band['_250m_16_days_EVI'].where(good_quality)
            else:
                ndvi = ds_band['_250m_16_days_NDVI']
                evi = ds_band['_250m_16_days_EVI']

            ds_local = xr.Dataset({'ndvi': ndvi, 'evi': evi})

            df_band = ds_local.to_dataframe().reset_index()
            df_band = df_band.dropna(subset=['ndvi', 'evi'], how='all')

            if df_band.empty:
                continue

            # Rename NetCDF coordinate columns to match grid builder expectations
            df_band = df_band.rename(columns={'lat': 'latitude', 'lon': 'longitude'})

            # Format time coordinates
            df_band['time'] = pd.to_datetime([str(t) for t in df_band['time']])
            df_band = df_band.rename(columns={'time': 'valid_time'})

            # Map coordinates to 5km master grid cells
            df_mapped = grid.assign_nearest_cell(df_band, lon_col='longitude', lat_col='latitude')

            # Aggregate 250m pixels into 5km cells
            df_grouped = df_mapped.groupby(['cell_id', 'valid_time'])[['ndvi', 'evi']].mean().reset_index()

            if df_grouped.empty:
                continue

            if not df_grouped.empty:
                all_processed_frames.append(df_grouped)

        log(f"Finished year {year}. Memory and swap fully preserved.")

    log("\nConsolidating and deduplicating global MODIS records...")
    master_modis = pd.concat(all_processed_frames, ignore_index=True)
    master_modis = master_modis.groupby(['cell_id', 'valid_time'])[['ndvi', 'evi']].mean().reset_index()

    log("Running daily forward-fill expansion...")
    master_modis = master_modis.sort_values(['cell_id', 'valid_time'])
    
    expanded_frames = []
    for cell_id, group in master_modis.groupby('cell_id'):
        group = group.set_index('valid_time').resample('1D').ffill().reset_index()
        group['cell_id'] = cell_id
        expanded_frames.append(group)
        
    final_modis_df = pd.concat(expanded_frames, ignore_index=True)

    output_path = Path("data/processed/modis_daily_features_5km.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_modis_df.to_csv(output_path, index=False)

    log(f"\nSuccessfully processed and saved daily MODIS features to: {output_path}")

if __name__ == "__main__":
    main()