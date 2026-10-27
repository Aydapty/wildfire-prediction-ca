import sys
import os
from pathlib import Path
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import log

def diagnose_duplicates():
    processed_dir = Path("data/processed")
    
    log("Checking individual component files for duplicates...")
    
    # 1. Master Grid check
    master_grid = pd.read_csv(processed_dir / "master_grid_5km.csv")
    grid_dups = master_grid.duplicated(subset=['cell_id']).sum()
    log(f"Master Grid duplicates (cell_id): {grid_dups}")
    
    # 2. ERA5 check
    era5 = pd.read_csv(processed_dir / "era5_daily_features_5km.csv")
    era5_dups = era5.duplicated(subset=['cell_id', 'valid_time']).sum()
    log(f"ERA5 duplicates (cell_id, valid_time): {era5_dups}")
    
    # 3. MODIS check
    modis = pd.read_csv(processed_dir / "modis_daily_features_5km.csv")
    modis_dups = modis.duplicated(subset=['cell_id', 'valid_time']).sum()
    log(f"MODIS duplicates (cell_id, valid_time): {modis_dups}")
    
    # 4. FPA-FOD check
    fpa_fod = pd.read_csv(processed_dir / "fpa_fod_mapped_5km.csv")
    fpa_dups = fpa_fod.duplicated(subset=['cell_id', 'valid_time']).sum()
    log(f"FPA-FOD duplicates (cell_id, valid_time): {fpa_dups}")

if __name__ == "__main__":
    diagnose_duplicates()