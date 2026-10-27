import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import log

def check_duplicates():
    log("Loading master training dataset...")
    df = pd.read_parquet("data/processed/master_training_dataset.parquet")
    
    log(f"Total rows in dataset: {len(df):,}")

    # 1. Check for duplicate spatiotemporal keys (the core panel grain)
    key_duplicates = df.duplicated(subset=['cell_id', 'valid_time']).sum()
    log(f"Duplicate (cell_id, valid_time) entries: {key_duplicates}")

    # 2. Check for completely identical rows across all columns
    row_duplicates = df.duplicated().sum()
    log(f"Completely identical duplicate rows: {row_duplicates}")

    if key_duplicates == 0 and row_duplicates == 0:
        log("\nResult: PASSED. The dataset is 100% duplicate-free at both the key and row levels.")
    else:
        log("\nResult: WARNING. Duplicates were found.")

if __name__ == "__main__":
    check_duplicates()