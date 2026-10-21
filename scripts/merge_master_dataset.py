import os
import sys
from pathlib import Path
from datetime import datetime
import gc
import pandas as pd

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def log(msg):
    """Helper function to print messages with a precise timestamp."""
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    print(f"{timestamp} {msg}")

def downcast_dtypes(df):
    """Optimize memory by downcasting float64/int64 to smaller types."""
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def main():
    log("Loading processed feature files with memory optimization...")
    processed_dir = Path(PROJECT_ROOT)/"data"/"processed"

    # Load individual components and downcast
    log("Loading master_grid...")
    master_grid = pd.read_csv(processed_dir/"master_grid_5km.csv")
    master_grid = downcast_dtypes(master_grid)

    log("Loading era5...")
    era5 = pd.read_csv(processed_dir/"era5_daily_features_5km.csv")
    # Drop native ERA5 coordinates since cell_id already maps to master_grid
    era5 = era5.drop(columns=['latitude', 'longitude'], errors='ignore')
    era5 = downcast_dtypes(era5)

    log("Loading modis...")
    modis = pd.read_csv(processed_dir/"modis_daily_features_5km.csv")
    modis = modis.drop(columns=['latitude', 'longitude'], errors='ignore')
    modis = downcast_dtypes(modis)

    log("Loading fpa_fod...")
    fpa_fod = pd.read_csv(processed_dir/"fpa_fod_mapped_5km.csv")
    # Drop raw point-level coordinates since cell_id maps to master_grid centroids
    fpa_fod = fpa_fod.drop(columns=['LATITUDE', 'LONGITUDE'], errors='ignore')
    # Ensure fpa_fod is strictly unique per cell and day to prevent row multiplication
    if 'fire_occurred' not in fpa_fod.columns:
        fpa_fod['fire_occurred'] = 1
    fpa_fod = fpa_fod.groupby(['cell_id', 'valid_time'], as_index=False)['fire_occurred'].max()
    fpa_fod = downcast_dtypes(fpa_fod)

    # Ensure data formats match for clean merging
    log("Formatting timestamps for era5...")
    era5['valid_time'] = pd.to_datetime(era5['valid_time'])
    log("Formatting timestamps for modis...")
    modis['valid_time'] = pd.to_datetime(modis['valid_time'])
    log("Formatting timestamps for fpa_fod...")
    fpa_fod['valid_time'] = pd.to_datetime(fpa_fod['valid_time'])

    # Merge ERA5 and MODIS layers
    log("Merging era5 and modis layers to df_merged...")
    df_merged = pd.merge(era5, modis, on=['cell_id', 'valid_time'], how='outer')

    # Free up memory from raw dataframes no longer needed
    del era5, modis
    gc.collect()

    # Merge static grid metadata (lat, lon, elevation/topography if present)
    log("Merging df_merged and master_grid layers to df_merged...")
    df_merged = pd.merge(df_merged, master_grid, on='cell_id', how='left')
    del master_grid
    gc.collect()

    # Merge fire ignition labels (left join so non-fire days/cells correctly show 0 or NaN target)
    log("Merging df_merged and fpa_fod layers to df_merged...")
    df_merged = pd.merge(df_merged, fpa_fod, on=['cell_id', 'valid_time'], how='left')
    del fpa_fod
    gc.collect()

    # Clean up target labels (fill missing ignition flags with 0 if appropriate)
    log("Cleaning target labels (filling missing ignitions flags with 0 if appropriate)...")
    if 'fire_occurred' in df_merged.columns:
        df_merged['fire_occurred'] = df_merged['fire_occurred'].fillna(0)

    # Clean up duplicate coordinate columns if they were created during merges
    log("Cleaning up duplicate coordinate columns...")
    if 'latitude_y' in df_merged.columns and 'latitude_x' in df_merged.columns:
        df_merged = df_merged.rename(columns={'latitude_y': 'latitude'})
        df_merged = df_merged.drop(columns=['latitude_x'])
        
    if 'longitude_y' in df_merged.columns and 'longitude_x' in df_merged.columns:
        df_merged = df_merged.rename(columns={'longitude_y': 'longitude'})
        df_merged = df_merged.drop(columns=['longitude_x'])

    # Sort chronologically and spatially
    log("Sorting df_merged chronologically and spatially...")
    df_merged = df_merged.sort_values(['valid_time', 'cell_id']).reset_index(drop=True)

    output_path = processed_dir/"master_training_dataset.parquet"
    log(f"Saving unified training dataset to: {output_path}")
    df_merged.to_parquet(output_path, index=False)

    log(f"Done! Final dataset shape: {df_merged.shape}")

if __name__ == "__main__":
    main()