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

# File paths
era5_root_dir = os.path.join(PROJECT_ROOT, "data", "raw", "ecmwf_era5_ranalysis")
master_grid_path = os.path.join(PROJECT_ROOT, "data", "processed", "master_grid_5km.csv")
processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
output_path = os.path.join(processed_dir, "era5_daily_features_5km.csv")

years = ["2022", "2023", "2024"]

def calculate_relative_humidity(temp_k, dewpoint_k):
    """Calculates relative humdity (%) from temperature and dewpoint in Kevlin"""
    temp_c = temp_k - 273.15
    dew_c = dewpoint_k - 273.15

    # Magnus formula approximation for relative humidity
    rh = 100 * (np.exp((17.625 * dew_c) / (243.04 + dew_c)) / np.exp((17.625 * temp_c) / (243.04 + temp_c)))

    return np.clip(rh, 0, 100)

def main():
    print("Loading master spatial grid anchor...")
    grid = SpatialGrid(master_grid_path)
    grid_df = grid.get_grid()

    all_daily_features = []

    # Loop through each year directory
    for year in years:
        year_dir = os.path.join(era5_root_dir, year)
        if not os.path.exists(year_dir):
            print(f"Skipping {year}: Directory not found.")
            continue

        netcdf_files = [os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith('.nc')]

        if not netcdf_files:
            print(f"Warning: No NetCDF files found in {year_dir}.")
            continue

        print(f"Processing year {year} with {len(netcdf_files)} files...")

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

        # Convert to DataFrame for spatial mapping
        df_era5 = ds_daily.to_dataframe().reset_index()

        if 'longitude' in df_era5.columns and 'latitude' in df_era5.columns:
            print(f"Mapping ERA5 coordinates to 5km grid cells for {year}...")
            df_era5_mapped = grid.assign_nearest_cell(df_era5, lon_col='longitude', lat_col='latitude')
            all_daily_features.append(df_era5_mapped)

        # Close datasets
        for d in datasets:
            d.close()
        ds.close()

    if all_daily_features:
        master_era5_df = pd.concat(all_daily_features, ignore_index=True)
        os.makedirs(processed_dir, exist_ok=True)
        master_era5_df.to_csv(output_path, index=False)
        print(f"Successfully processed and saved daily ERA5 features to: {output_path}")
    else:
        print("No valid spatial features were extracted from the ERA5 files.")

if __name__ == "__main__":
    main()