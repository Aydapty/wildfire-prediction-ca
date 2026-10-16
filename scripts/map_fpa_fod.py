import os
import sys
import sqlite3
import pandas as pd

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.grid_builder import SpatialGrid

# File paths
sqlite_path = os.path.join(PROJECT_ROOT, "data", "raw", "fpa_fod", "FPA_FOD_20260615.sqlite")
master_grid_path = os.path.join(PROJECT_ROOT, "data", "processed", "master_grid_5km.csv")
processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
output_path = os.path.join(processed_dir, "fpa_fod_mapped_5km.csv")

def main():
    print("Loading master spatial grid anchor...")
    grid = SpatialGrid(master_grid_path)

    print(f"Connecting to SQLite database: {sqlite_path}")
    conn = sqlite3.connect(sqlite_path)

    # Query only California records to optimize memory and processing
    print("Querying California wildfire records form SQLite...")
    query = """
    SELECT FPA_ID, FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_DOY, 
    LATITUDE, LONGITUDE, FIRE_SIZE, FIRE_SIZE_CLASS, 
    NWCG_CAUSE_CLASSIFICATION, STATE, COUNTY
    FROM Fires
    Where STATE = 'CA'
    """

    df_fires = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(df_fires)} California fire records.")

    print("Mapping fire coordinates to nearest 5km grid cells...")
    df_fires_mapped = grid.assign_nearest_cell(
        df=df_fires,
        lon_col="LONGITUDE",
        lat_col="LATITUDE"
    )

    # Convert Julian dates to standard date format for consistent merging with ERA5 and MODIS layers
    df_fires_mapped['DISCOVERY_DATE'] = pd.to_datetime(df_fires_mapped['DISCOVERY_DATE'], unit='D', origin='julian')

    # Standardize 'DISCOVERY_DATE' data column to 'valid_time' for consistent merging with ERA5 and MODIS layers
    df_fires_mapped = df_fires_mapped.rename(columns={'DISCOVERY_DATE': 'valid_time'})

    # Filter records to align with the 2022-2024 feature baseline window (ERA5 & MODIS)
    df_fires_mapped = df_fires_mapped[
        (df_fires_mapped['valid_time'] >= '2022-01-01') &
        (df_fires_mapped['valid_time'] <= '2024-12-31')
    ]

    # Save processed output
    os.makedirs(processed_dir, exist_ok=True)
    df_fires_mapped.to_csv(output_path, index=False)
    print(f"Successfully mapped and saved fire records to: {output_path}")

if __name__ == "__main__":
    main()