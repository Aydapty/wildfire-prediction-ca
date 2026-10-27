# Core Source Code (`src/`)

Modular, reusable Python packages and class definitions powering the California wildfire prediction pipeline.

## Modules

- **`grid_builder.py`**: Contains the `SpatialGrid` class responsible for initializing the fixed 5 km spatial anchor, managing cell indexing (`cell_id`), and performing spatial lookups via `scipy.spatial.KDTree` to map external datasets to nearest grid coordinates.
- **`utils.py`**: Provides a modular logging utility (`log`) that prefixes console outputs with precise timestamps for tracking pipeline execution progress[cite: 7].

## Boilerplate Import Setup

Because standalone scripts reside in `scripts/` while core modules live in `src/`, include the following path configuration at the top of your execution scripts to import `src` modules correctly:

```python
import os
import sys

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import log
from src.grid_builder import SpatialGrid