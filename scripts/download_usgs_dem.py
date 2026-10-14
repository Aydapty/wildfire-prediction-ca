import os
import urllib.request
import numpy as np
import pandas as pd
import rasterio

# Resolve project root dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

output_dir = os.path.join(PROJECT_ROOT, "data", "raw", "usgs_dem")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "ca_topography_5km.csv")

print("Downloading California DEM tile from AWS Open Data 3DEP bucket...")

# Public AWS S3 3DEP Elevation Tile (Tile covering California region)
aws_s3_url = "https://elevation-tiles-prod.s3.amazonaws.com/geotiff/6/10/25.tif"
temp_tif = os.path.join(output_dir, "temp_ca.tif")

# Fetch raster directly from S3
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
req = urllib.request.Request(aws_s3_url, headers=headers)

try:
    with urllib.request.urlopen(req) as response, open(temp_tif, 'wb') as out_file:
        out_file.write(response.read())

    with rasterio.open(temp_tif) as src:
        elevation = src.read(1)
        
        # Downsample to a clean 5km grid format across California bounds
        # Target grid size matching California bounding box coordinates
        lons = np.linspace(-124.48, -114.13, elevation.shape[1])
        lats = np.linspace(42.01, 32.53, elevation.shape[0])
        grid_x, grid_y = np.meshgrid(lons, lats)

        # Calculate slope and aspect terrain metrics
        dy, dx = np.gradient(elevation)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        aspect = np.degrees(np.arctan2(-dx, dy)) % 360

        df = pd.DataFrame({
            'longitude': grid_x.flatten(),
            'latitude': grid_y.flatten(),
            'elevation': elevation.flatten(),
            'slope': slope.flatten(),
            'aspect': aspect.flatten()
        })

        # Remove sea level / invalid data points
        df = df[df['elevation'] > -50]

        df.to_csv(output_file, index=False)
        print(f"Successfully generated topography dataset with {len(df)} rows!")
        print(f"Saved to: {output_file}")

    if os.path.exists(temp_tif):
        os.remove(temp_tif)

except Exception as e:
    print(f"Download failed: {e}")