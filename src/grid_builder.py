import os
import pandas as pd
import numpy as np

class SpatialGrid:
    """ Standardized 5km spatial grid reference anchor for California wildfire datasets """

    def __init__(self, topo_path: str):
        """ Load the baseline 5km DEM grid coordinates and terrain metrics """
        if not os.path.exists(topo_path):
            raise FileNotFoundError(f"Topography baseline not found at: {topo_path}")

        self.grid_df = pd.read_csv(topo_path)

        # Generate a unique integer cell_id for fast spatial indexing
        self.grid_df["cell_id"] = np.arange(len(self.grid_df))

        # Reorder columns to place cell_id first
        cols = ["cell_id", "longitude", "latitude", "elevation", "slope", "aspect"]
        self.grid_df = self.grid_df[cols]

    def get_grid(self) -> pd.DataFrame:
        """ Return a copy of the base spatial grid """
        return self.grid_df.copy()

    def assign_nearest_cell(self, df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
        """ Map arbitrary lat/lon points (e.g., FPA-FOD fire ignitions) to the nearest 5km grid cell_id """
        grid_coords = self.grid_df[["longitude", "latitude"]].values
        target_coords = df[[lon_col, lat_col]].values

        # Compute minimum Euclidean distance in coordinate space for fast lookup
        from scipy.spatial import KDTree

        tree = KDTree(grid_coords)
        _, indices = tree.query(target_coords)

        df_mapped = df.copy()
        df_mapped["cell_id"] = self.grid_df.iloc[indices]["cell_id"].values
        return df_mapped

if __name__ == "__main__":
    # Test script execution
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    topo_csv = os.path.join(project_root, "data", "raw", "usgs_dem", "ca_topography_5km.csv")

    grid = SpatialGrid(topo_csv)
    base_df = grid.get_grid()

    print(f"Master Grid Intialized Successfully")
    print(f"Total 5km Spatial Cells: {len(base_df)}")
    print("\nSample Grid Rows:")
    print(base_df.head())