# transform_master_dataset_breakdown

The following is a breakdown of what transform_master_dataset.py does as a script, what it results in, and why it is needed. I (aydapty/Aditya Pundhir) added this since I felt like it would clarify a lot of things. This group project requires us to use CNN and other related models (plus they're strong models too), thus we needed to transform our current work into a workflow that could accommodate as such.

### 1. Zero-RAM Partition Filtering

Instead of reading all 83.4 million rows into memory, PyArrow uses predicate pushdown filters to scan only the byte blocks belonging to the specific year being processed (2022 for train, 2023 for validation, 2024 for testing).

### 2. Spatial Canvas Reconstitution

For every single day in the split, the script reconstructs California’s physical map. It creates a blank 2D canvas of shape (403 rows, 352 cols) -matching the discrete bounding box of the state’s 5km cells. It then maps the 9 environmental features into this matrix:
* 4 weather layers (`t2m, rh2m, wind_speed_10m, tp`)
* 2 vegetation layers (`ndvi, evi`)
* 3 topography layers (`elevation, slope, aspect`)

### 3. Border Zero-Padding

When evaluating cells along the Pacific coastline or the state boundary, slicing a window around the cell would normally reach out of bounds. The script adds a boundary padding of radius $r=3$(mode="constant", constant_values=0), ensuring every cell has a complete neighborhood.

### 4. Prior-Day Lagged Patch Extraction (t-1 -> t)

To enforce zero temporal leakage, the script looks at the full (9, H, W) surface from yesterday (t-1) to predict whether a fire ignited at the center cell today (t).
* Around the target cell's $(x, y)$ coordinate, it cuts out a $(2r+1) \times (2r+1) = 7 \times 7$ grid window.
* Because each grid cell is 5km, that $7 \times 7$ patch represents a physical $35 km \times 35 km$ regional box centered directly on the target location.

### 5. Output Array Generation

The resulting patches and labels are saved as `.npy` arrays:
* `X_train.npy` / `y_train.npy`: 2022 split with shape `(557616, 9, 7, 7)` and `(557616,)`. Contains 100% of fires downsampled with 2% non-fire background points for efficient gradient optimization.
* `X_val.npy` / `y_val.npy`: 2023 split with shape `(1388899, 9, 7, 7)` and `(1388899,)`. Evaluated against a denser 5% non-fire background sample to provide realistic, rigorous PR-AUC and ROC-AUC validation benchmarks.
* `X_test.npy` / `y_test.npy`: 2024 split with shape `(1392854, 9, 7, 7)` reserved strictly for final out-of-sample evaluation.

### Example CNN processing of the resulting tensors:

```text
Input Tensor: (Batch, 9, 7, 7)
               │
               ▼
  [ Conv2D Layer (e.g., 3x3 kernels) ]
  Slides across the 7x7 spatial patch, computing dot products
  across all 9 channels simultaneously.
               │
               ▼
  [ Spatial Feature Maps ]
  Detects:
   - Elevation gradients (steep canyons vs. flat valleys)
   - Moisture corridors (dry chaparral bordered by moist vegetation)
   - Atmospheric fronts (high wind speeds blowing toward hot, dry cells)
               │
               ▼
  [ Pooling & Dense Classification Head ]
               │
               ▼
  Prediction: Ignition Probability (0.0 to 1.0)
```

## Info about Tensor Shape `(samples, 9, 7, 7)`

NOTE: A $403 \times 352$ grid represents the bounding box rectangle enclosing all 5km coordinate points in California!

### Input Channels (`C = 9`) Replace RGB

A standard image uses 3 color channels (Red, Green, Blue). A convolutional kernel operates across all input channels at once. When a 2D filter slides over your patch, it calculates interactions across temperature, humidity, wind, slope, and vegetation simultaneously.

Contains the 9 physical and environmental variables:

* `t2m` (2-meter air temperature)
* `rh2m` (2-meter relative humidity)
* `wind_speed_10m` (10-meter surface wind speed)
* `tp` (total precipitation)
* `ndvi` (Normalized Difference Vegetation Index / live green canopy density)
* `evi` (Enhanced Vegetation Index / structural canopy greenness)
* `elevation` (topographical height)
* `slope` (terrain incline)
* `aspect` (azimuth/direction slope faces relative to sun and wind)

### Spatial Gradients (`H = 7, W = 7`)

Convolutions compute spatial derivatives (differences between adjacent pixels). In wildfire behavior, ignition risk is driven by spatial contrasts—such as wind funneling through a narrow canyon slope or an arid perimeter next to dense fuel biomass. The $7 \times 7$ (35 km $\times$ 35km) window gives the convolutional kernels the exact neighborhood context needed to detect those spatial patterns. 

NOTE: Each cell is 5 km $\times$ 5 km!

### Latitude breakdown (403 rows)

* California spans from roughly $32.5^\circ\text{ N}$ (Mexican border) to $42.0^\circ\text{ N}$ (Oregon border).
* Total span: $\approx 9.5^\circ$ of latitude.
* At your $\approx 5\text{ km}$ spatial resolution (where $1^\circ \text{ lat} \approx 111\text{ km}$, so $5\text{ km} \approx 0.045^\circ / \text{step}$):

$$\frac{9.5^\circ}{0.045^\circ} \approx \mathbf{403 \text{ discrete latitude rows}}$$

### Longitude (352 columns)

* California spans from roughly $-124.4^\circ\text{ W}$ (Cape Mendocino) to $-114.1^\circ\text{ W}$ (Colorado River border).
* Total span: $\approx 10.3^\circ$ of longitude.
* At California's latitudes ($1^\circ \text{ lon} \approx 85\text{ km}$ to $95\text{ km}$, so $5\text{ km} \approx 0.055^\circ / \text{step}$):

$$\frac{10.3^\circ}{0.055^\circ} \approx \mathbf{352 \text{ discrete longitude columns}}$$

## Important next steps

* Running `transformer_master_dataset.py` will create the resulting tensors and store them in `data/processed/tensors/`. 
* The next step is to run `src/dataset.py` to create `WildfireTensorDataset`. 
* Our team will then load up `notebooks/cnn_baseline.ipynb` to run training and validation loops.