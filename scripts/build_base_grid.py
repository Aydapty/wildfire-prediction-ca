import os
import sys

# Dynamically resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

import pandas as pd
from src.grid_builder import SpatialGrid

raw_topo_csv = os.path.join(PROJECT_ROOT, "data", "raw", "usgs_dem", "ca_topography_5km.csv")
processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)
output_master_grid = os.path.join(processed_dir, "master_grid_5km.csv")

def main():
    print("Initializing SpatialGrid anchor from raw USGS DEM data...")
    grid = SpatialGrid(raw_topo_csv)
    master_df = grid.get_grid()

    # Save out the official baseline grid anchor
    master_df.to_csv(output_master_grid, index=False)
    print(f"Master 5km Grid Anchor successfully created with {len(master_df)} cells.")
    print(f"Saved to: {output_master_grid}")

if __name__ == "__main__":
    main()