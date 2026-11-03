# wildfire-prediction-ca
COSC 4337 Data Science II (Fall 2026) | University of Houston\
Group 24 Project: Machine learning pipeline predicting daily 24-hour wildfire ignition probabilities across California grid cells using multi-modal environmental data.

---

## Repository Structure

* **`data/`**: Manages raw datasets and processed features (see [`data/README.md`](data/README.md) for full schema details).
* **`docs/`**: Project documentation and architecture notes.
* **`notebooks/`**: Exploratory data analysis and model prototyping notebooks.
* **`scripts/`**: Executable data acquisition, processing, and merge scripts (see [`scripts/README.md`](scripts/README.md)).
* **`src/`**: Reusable core source code and spatial utilities (see [`src/README.md`](src/README.md)).

## Quickstart: Loading the Master Dataset

The preprocessed 2022–2024 spatiotemporal training panel is compiled into an optimized Apache Parquet file. Teammates can load it directly in Python for modeling:

```python
import pandas as pd

# Load the consolidated daily panel (~1.38 GB)
df = pd.read_parquet("data/processed/master_training_dataset.parquet")
print(df.info())
```

## Pipeline Reproduction

To regenerate the processed features and master dataset from scratch, run the scripts in the following order (make sure to download raw data first with steps mentioned in [`data/README.md`](data/README.md)):

```bash
# 1. Build spatial grid and extract DEM baseline
python scripts/build_base_grid.py

# 2. Process individual environmental and target layers
python scripts/process_era5.py
python scripts/process_modis.py
python scripts/map_fpa_fod.py

# 3. Consolidate into the final compressed training panel
python scripts/merge_master_dataset.py

---

## Model Evaluation and Testing

## Overview

This part of the project evaluates three wildfire prediction models:

-   Logistic Regression
-   Random Forest
-   XGBoost

It compares their performance using 2023 validation data and unseen 2024
test data. The goal is to assess how effectively each model identifies
wildfire observations and how consistently it performs across years.

## Data Split

The project follows a chronological split:

  Year   Purpose
  ------ ---------------------------
  2022   Model training
  2023   Validation and evaluation
  2024   Final testing

Decision thresholds were selected using the 2023 validation data and
kept fixed for the 2024 final test. The 2024 results were not used to
tune the thresholds.

## Models and Decision Thresholds

  Model                   Decision threshold
  --------------------- --------------------
  Logistic Regression                  0.018
  Random Forest                        0.180
  XGBoost                              0.690

A prediction is classified as positive when its predicted probability
meets or exceeds the model's threshold.

## Prediction Files

The notebook uses pre-generated prediction files for both periods.

**2023 validation predictions** -
`logreg_validation_predictions.parquet` -
`rf_validation_predictions.parquet` -
`xgb_validation_predictions.parquet`

**2024 final test predictions** -
`logreg_test_2024_predictions.parquet` -
`rf_test_2024_predictions.parquet` - `xgb_test_2024_predictions.parquet`

Update the file paths in the notebook to match the folders on your
computer. The notebook loads only the `y_true` and `y_prob` columns
needed for evaluation to help reduce memory usage.

## Evaluation Metrics

-   **Precision:** The proportion of positive predictions that are
    actual wildfire observations.
-   **Recall:** The proportion of actual wildfire observations correctly
    identified by the model.
-   **F1-score:** The harmonic mean of precision and recall.
-   **ROC-AUC:** Measures how well the model distinguishes positive and
    negative observations across classification thresholds.
-   **Average Precision (AP):** Summarizes performance across the
    precision-recall curve.

Wildfire observations are rare compared with non-wildfire observations,
so accuracy alone can be misleading. The notebook therefore focuses on
precision, recall, F1-score, ROC-AUC, and Average Precision.

## Visualizations

The notebook includes:

-   Precision, recall, and F1-score comparisons
-   Confusion matrices
-   ROC curves
-   Precision-recall curves
-   Comparisons of model performance between 2023 and 2024

These visualizations illustrate the trade-off between detecting wildfire
observations and generating false alarms.

## Main Results

Random Forest achieved the strongest overall performance in both
evaluation periods. It had the highest F1-score, ROC-AUC, and Average
Precision, indicating the best overall balance among the three models.

XGBoost achieved the highest recall in both years, meaning it identified
the largest proportion of actual wildfire observations. However, it also
produced more false positives than Random Forest.

Logistic Regression performed substantially worse than the other two
models.

### 2024 Final Test Results

  Model                   Precision   Recall   F1-score   ROC-AUC   Average Precision
  --------------------- ----------- -------- ---------- --------- -------------------
  Logistic Regression        0.0280   0.0398     0.0329    0.9657              0.0120
  Random Forest              0.4149   0.5472     0.4719    0.9910              0.4458
  XGBoost                    0.3329   0.5587     0.4172    0.9458              0.2704

Overall, Random Forest is the recommended model when balancing precision
and recall is the priority. XGBoost is an alternative when maximizing
wildfire detection is more important.

## How to Run

1.  Make sure the six prediction Parquet files listed above are
    available.
2.  Open the model evaluation notebook.
3.  Update the prediction file paths to match your local folders.
4.  Install the required packages if they are not already available:
    -   `pandas`
    -   `numpy`
    -   `scikit-learn`
    -   `matplotlib`
    -   `fastparquet`
    -   `xgboost` if needed by the notebook's imports
5.  Run the notebook cells in order.

The notebook reads Parquet files using the `fastparquet` engine.

## Scope of This Work

This notebook focuses on evaluating and comparing prediction outputs
generated by the trained models. It does not train the models. Model
training and prediction-file generation are handled separately.

