import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.grid_builder import SpatialGrid

def main():
    print("Loading master spatial grid anchor...")
    master_grid_path = os.path.join(PROJECT_ROOT, "data", "processed", "master_grid_5km.csv")
    grid = SpatialGrid(master_grid_path)
    modis_dir = Path("data/raw")

    # Find the MODIS NetCDF file recursively across subfolders
    modis_files = list(modis_dir.glob("**/MOD13Q1*.nc"))
    if not modis_files:
        raise FileNotFoundError("No MODIS NetCDF file found in data/raw/")

    modis_path = modis_files[0]
    print(f"Opening MODIS dataset lazily: {modis_path.name}")

    ds = xr.open_dataset(modis_path)
    print("Dataset variables available:", list(ds.variables))

    # Extract unique years and filter strictly for 2022-2024
    time_vals = ds['time'].values
    all_years = sorted(list(set(pd.to_datetime([str(t) for t in time_vals]).year)))
    years = [y for y in all_years if 2022 <= y <= 2024]
    print(f"Detected target years to process: {years}")

    output_path = Path("data/processed/modis_daily_features_5km.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_chunk = True

    for year in years:
        print(f"\n--- Processing Year {year} ---")
        ds_year = ds.sel(time=str(year))

        lats = ds_year['lat'].values
        chunk_size = 50  # Number of latitude rows per micro-chunk

        for i in range(0, len(lats), chunk_size):
            lat_slice = lats[i:i + chunk_size]
            print(f"  -> Processing latitude band {i}/{len(lats)}...")

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

            # Perform daily temporal forward-fill expansion per cell
            df_grouped = df_grouped.sort_values(['cell_id', 'valid_time'])
            
            expanded_frames = []
            for cell_id, group in df_grouped.groupby('cell_id'):
                group = group.set_index('valid_time').resample('1D').ffill().reset_index()
                group['cell_id'] = cell_id
                expanded_frames.append(group)
                
            df_band_daily = pd.concat(expanded_frames, ignore_index=True)

            # Append incrementally to CSV
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            df_band_daily.to_csv(output_path, mode=mode, header=header, index=False)

            first_chunk = False

        print(f"Finished year {year}. Memory and swap fully preserved.")

    print(f"\nSuccessfully processed and saved daily MODIS features to: {output_path}")

if __name__ == "__main__":
    main()