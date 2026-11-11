import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from tqdm import tqdm

# Set project root to current repository
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.utils import log
except ImportError:
    def log(msg): print(f"[*] {msg}")

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROCESSED_DIR / "tensors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_RADIUS = 3  # (2 * 3) + 1 = 7x7 spatial patch
FEATURE_COLS = [
    "t2m", "rh2m", "wind_speed_10m", "tp",
    "ndvi", "evi",
    "elevation", "slope", "aspect"
]
TARGET_COL = "fire_occurred"
LAT_COL = "latitude"
LON_COL = "longitude"
LOAD_COLS = ["valid_time", LAT_COL, LON_COL, TARGET_COL] + FEATURE_COLS


def build_grid_index(df_coords):
    unique_lats = np.sort(df_coords[LAT_COL].unique())[::-1]
    unique_lons = np.sort(df_coords[LON_COL].unique())
    lat_to_row = {lat: idx for idx, lat in enumerate(unique_lats)}
    lon_to_col = {lon: idx for idx, lon in enumerate(unique_lons)}
    grid_shape = (len(unique_lats), len(unique_lons))
    return lat_to_row, lon_to_col, grid_shape


def compute_train_scaler(df_train_sample):
    stats = {}
    for col in FEATURE_COLS:
        mean_val = float(df_train_sample[col].mean())
        std_val = float(df_train_sample[col].std())
        stats[col] = (mean_val, std_val if std_val > 0 else 1.0)
    return stats


def extract_daily_tensors(df_split, lat_to_row, lon_to_col, grid_shape, stats, r=WINDOW_RADIUS, sample_neg_frac=None):
    for col in FEATURE_COLS:
        m, s = stats[col]
        df_split[col] = (df_split[col] - m) / s

    all_dates = sorted(df_split["valid_time"].dt.date.unique())
    patch_dim = 2 * r + 1
    num_channels = len(FEATURE_COLS)
    H, W = grid_shape
    
    X_list = []
    y_list = []

    for i in tqdm(range(1, len(all_dates)), desc="Processing Days", leave=False):
        prev_date = all_dates[i - 1]
        curr_date = all_dates[i]

        df_prev = df_split[df_split["valid_time"].dt.date == prev_date]
        df_curr = df_split[df_split["valid_time"].dt.date == curr_date]

        if df_prev.empty or df_curr.empty:
            continue

        spatial_plane = np.zeros((num_channels, H, W), dtype=np.float32)
        rows = df_prev[LAT_COL].map(lat_to_row).values
        cols = df_prev[LON_COL].map(lon_to_col).values
        for c_idx, col in enumerate(FEATURE_COLS):
            spatial_plane[c_idx, rows, cols] = df_prev[col].fillna(0.0).values

        padded_plane = np.pad(spatial_plane, ((0, 0), (r, r), (r, r)), mode="constant", constant_values=0)

        if sample_neg_frac is not None:
            pos_mask = df_curr[TARGET_COL] == 1.0
            neg_mask = df_curr[TARGET_COL] == 0.0
            df_curr = pd.concat([
                df_curr[pos_mask],
                df_curr[neg_mask].sample(frac=sample_neg_frac, random_state=42)
            ])

        target_rows = df_curr[LAT_COL].map(lat_to_row).values
        target_cols = df_curr[LON_COL].map(lon_to_col).values
        targets = df_curr[TARGET_COL].values

        for r_idx, c_idx, label in zip(target_rows, target_cols, targets):
            patch = padded_plane[:, r_idx : r_idx + patch_dim, c_idx : c_idx + patch_dim]
            X_list.append(patch)
            y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def main():
    parquet_path = PROCESSED_DIR / "master_training_dataset.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}. Verify path.")

    dataset = ds.dataset(parquet_path, format="parquet")

    log("Building grid coordinate index...")
    df_coords = dataset.to_table(columns=[LAT_COL, LON_COL]).to_pandas().drop_duplicates()
    lat_to_row, lon_to_col, grid_shape = build_grid_index(df_coords)
    log(f"Grid Lattice: {grid_shape[0]} rows x {grid_shape[1]} cols")
    del df_coords

    t_2022_start = pa.scalar(pd.Timestamp("2022-01-01"), type=pa.timestamp("ns"))
    t_2022_end   = pa.scalar(pd.Timestamp("2022-12-31 23:59:59"), type=pa.timestamp("ns"))
    t_2023_start = pa.scalar(pd.Timestamp("2023-01-01"), type=pa.timestamp("ns"))
    t_2023_end   = pa.scalar(pd.Timestamp("2023-12-31 23:59:59"), type=pa.timestamp("ns"))
    t_2024_start = pa.scalar(pd.Timestamp("2024-01-01"), type=pa.timestamp("ns"))
    t_2024_end   = pa.scalar(pd.Timestamp("2024-12-31 23:59:59"), type=pa.timestamp("ns"))

    # 1. 2022 Train Split
    log("Loading 2022 Train partition...")
    filter_2022 = (ds.field("valid_time") >= t_2022_start) & (ds.field("valid_time") <= t_2022_end)
    df_train = dataset.to_table(columns=LOAD_COLS, filter=filter_2022).to_pandas()
    train_stats = compute_train_scaler(df_train)

    log("Extracting 2022 Train tensors...")
    X_train, y_train = extract_daily_tensors(
        df_train, lat_to_row, lon_to_col, grid_shape, train_stats, sample_neg_frac=0.02
    )
    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    log(f"Train saved: X={X_train.shape}, y={y_train.shape}")
    del df_train, X_train, y_train

    # 2. 2023 Val Split
    log("Loading 2023 Val partition...")
    filter_2023 = (ds.field("valid_time") >= t_2023_start) & (ds.field("valid_time") <= t_2023_end)
    df_val = dataset.to_table(columns=LOAD_COLS, filter=filter_2023).to_pandas()

    log("Extracting 2023 Val tensors...")
    X_val, y_val = extract_daily_tensors(
        df_val, lat_to_row, lon_to_col, grid_shape, train_stats, sample_neg_frac=0.05
    )
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)
    log(f"Val saved: X={X_val.shape}, y={y_val.shape}")
    del df_val, X_val, y_val

    # 3. 2024 Test Split
    log("Loading 2024 Test partition...")
    filter_2024 = (ds.field("valid_time") >= t_2024_start) & (ds.field("valid_time") <= t_2024_end)
    df_test = dataset.to_table(columns=LOAD_COLS, filter=filter_2024).to_pandas()

    log("Extracting 2024 Test tensors...")
    X_test, y_test = extract_daily_tensors(
        df_test, lat_to_row, lon_to_col, grid_shape, train_stats, sample_neg_frac=0.05
    )
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    log(f"Test saved: X={X_test.shape}, y={y_test.shape}")
    del df_test, X_test, y_test

    log("Local extraction finished. Output arrays saved to data/processed/tensors/")

if __name__ == "__main__":
    main()