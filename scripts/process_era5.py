import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.grid_builder import SpatialGrid
from src.utils import log

# File paths
era5_root_dir = os.path.join(PROJECT_ROOT, "data", "raw", "ecmwf_era5_ranalysis")
master_grid_path = os.path.join(PROJECT_ROOT, "data", "processed", "master_grid_5km.csv")
processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
output_path = os.path.join(processed_dir, "era5_daily_features_5km.csv")

years = ["2022", "2023", "2024"]

def calculate_relative_humidity(temp_k, dewpoint_k):
    """Calculates relative humdity (%) from temperature and dewpoint in Kelvin"""
    temp_c = temp_k - 273.15
    dew_c = dewpoint_k - 273.15

    # Magnus formula approximation for relative humidity
    rh = 100 * (np.exp((17.625 * dew_c) / (243.04 + dew_c)) / np.exp((17.625 * temp_c) / (243.04 + temp_c)))

    return np.clip(rh, 0, 100)

def main():
    log("Loading master spatial grid anchor...")
    grid = SpatialGrid(master_grid_path)
    grid_df = grid.get_grid()

    # Pre-extract master grid coordinates and cell IDs for vectorized xarray interpolation
    grid_lons = grid_df['longitude'].values
    grid_lats = grid_df['latitude'].values
    grid_cell_ids = grid_df['cell_id'].values

    target_lons = xr.DataArray(grid_lons, dims="cell")
    target_lats = xr.DataArray(grid_lats, dims="cell")

    all_daily_features = []

    # Loop through each year directory
    for year in years:
        year_dir = os.path.join(era5_root_dir, year)
        if not os.path.exists(year_dir):
            log(f"Skipping {year}: Directory not found.")
            continue

        netcdf_files = [os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith('.nc')]

        if not netcdf_files:
            log(f"Warning: No NetCDF files found in {year_dir}.")
            continue

        log(f"Processing year {year} with {len(netcdf_files)} files...")

        # Open all netcdf files for this year and merge them along dimensions (like variables)
        datasets = [xr.open_dataset(f) for f in netcdf_files]
        ds = xr.merge(datasets)

        # Compute relative humidity if temp and dewpoint exist
        if 't2m' in ds and 'd2m' in ds:
            ds['rh2m'] = calculate_relative_humidity(ds['t2m'], ds['d2m'])

        # Compute 10m wind speed from u and v components if available
        if 'u10' in ds and 'v10' in ds:
            ds['wind_speed_10m'] = np.sqrt(ds['u10']**2 + ds['v10']**2)

        # Resample each variable with its appropriate aggregation function
        daily_vars = []

        if 't2m' in ds:
            daily_vars.append(ds['t2m'].resample(valid_time='1D').mean())
        if 'rh2m' in ds:
            daily_vars.append(ds['rh2m'].resample(valid_time='1D').mean())
        if 'wind_speed_10m' in ds:
            daily_vars.append(ds['wind_speed_10m'].resample(valid_time='1D').max())
        if 'tp' in ds:
            daily_vars.append(ds['tp'].resample(valid_time='1D').sum())

        # Merge the resampled variables back into a single daily dataset
        ds_daily = xr.merge(daily_vars)

        # Interpolate ERA5 directly to the master grid coordinates (fixes coverage & duplicates)
        log(f"Interpolating ERA5 data directly to master grid coordinates for {year}...")
        ds_daily_interp = ds_daily.interp(
            longitude=target_lons, 
            latitude=target_lats, 
            method="nearest"
        )

        # Assign cell_id as a coordinate along the cell dimension
        ds_daily_interp = ds_daily_interp.assign_coords(cell_id=("cell", grid_cell_ids))

        # Convert to DataFrame
        df_era5_mapped = ds_daily_interp.to_dataframe().reset_index()
        df_era5_mapped = df_era5_mapped.drop(columns=['cell'], errors='ignore')

        # Enforce strictly 1 row per cell_id and valid_time
        df_era5_mapped = df_era5_mapped.groupby(['cell_id', 'valid_time'], as_index=False).mean(numeric_only=True)
        
        all_daily_features.append(df_era5_mapped)

        # Close datasets
        for d in datasets:
            d.close()
        ds.close()

    if all_daily_features:
        master_era5_df = pd.concat(all_daily_features, ignore_index=True)

        # Fix any dimensional (number/expver) or spatial mapping duplicates
        log("Aggregating duplicates to ensure strictly one row per cell_id and valid_time...")
        master_era5_df = master_era5_df.groupby(['cell_id', 'valid_time'], as_index=False).mean(numeric_only=True)

        os.makedirs(processed_dir, exist_ok=True)
        master_era5_df.to_csv(output_path, index=False)
        log(f"Successfully processed and saved daily ERA5 features [{len(master_era5_df):,} rows] to: {output_path}")
    else:
        log("No valid spatial features were extracted from the ERA5 files.")

if __name__ == "__main__":
    main()