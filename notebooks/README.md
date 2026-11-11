# Jupyter Notebooks (`notebooks/`)

This directory houses the end-to-end modeling, feature ablation, benchmarking, and evaluation notebooks for the California daily wildfire ignition risk prediction project.

The experiments span classical tabular machine learning models (Logistic Regression, Random Forest, XGBoost), feature ablation studies, 2024 independent out-of-sample holdout testing, and spatial deep learning baselines (WildfireCNN).

---

## Notebook Directory Matrix

| Notebook | Focus Area | Data Inputs | Primary Outputs & Artifacts | Validation PR-AUC (2023) | Test PR-AUC (2024) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `cnn_baseline.ipynb` | Deep Learning Baseline | `data/processed/tensors/` (`X_train.npy`, `X_val.npy`) | `models/cnn_baseline_run2_seeded.pt`<br>`outputs/cnn_baseline_run2_seeded_validation_metrics.json`<br>`outputs/cnn_baseline_run2_seeded_diagnostics.png` | **0.9214** (Seed 42)<br>*0.9317 (Run 1)* | *Pending* |
| `model_experiments_logreg.ipynb` | Linear Tabular Baseline | `data/processed/master_training_dataset.parquet` | `models/logreg_provisional.joblib`<br>`outputs/logreg_validation_predictions.parquet`<br>`outputs/logreg_validation_predictions.csv` | **0.0105** | *Evaluated in Test NB* |
| `model_experiments_random_forest.ipynb` | Non-Linear Ensemble Baseline | `data/processed/master_training_dataset.parquet` | `models/random_forest_provisional.joblib`<br>`outputs/rf_validation_predictions.parquet`<br>`outputs/rf_validation_metrics.json`<br>`outputs/rf_feature_importance.csv`<br>`outputs/rf_threshold_analysis.csv` | **0.4493** | *Evaluated in Test NB* |
| `model_experiments_xgboost.ipynb` | Gradient Boosting Baseline | `data/processed/master_training_dataset.parquet` | `models/xgboost_provisional.joblib`<br>`outputs/xgb_validation_predictions.parquet`<br>`outputs/xgb_validation_metrics.json`<br>`outputs/xgb_feature_importance.csv`<br>`outputs/xgb_threshold_analysis.csv` | **0.2673** | *Evaluated in Test NB* |
| `model_experiments_ablation.ipynb` | Environmental Feature Ablation | `data/processed/master_training_dataset.parquet` | `models/rf_ablation_*.joblib`<br>`outputs/rf_ablation_*_validation.parquet`<br>`outputs/rf_ablation_2023_comparison.csv`<br>`outputs/rf_ablation_2024_comparison.csv` | Weather: `0.0003`<br>Weather+Veg: `0.0059`<br>Full (9 feats): `0.4493` | Weather: `0.0006`<br>Weather+Veg: `0.0103`<br>Full (9 feats): `0.4458` |
| `model_experiments_2024_test.ipynb` | Independent 2024 Holdout Test | `data/processed/master_training_dataset.parquet`<br>`models/*_provisional.joblib` | `outputs/*_test_2024_predictions.parquet`<br>`outputs/*_test_2024_metrics.json`<br>`outputs/model_comparison_2024.csv` | — | LogReg: `0.0120`<br>XGB: `0.2704`<br>RF: `0.4458` |

---

## Detailed Notebook Descriptions

### 1. `cnn_baseline.ipynb` — Spatial 2D-CNN Baseline
* **Goal:** Model daily wildfire ignition risk by utilizing 2D spatial context around target cells at day $t-1$ to predict ignitions at day $t$.
* **Input Schema:** Multi-modal 4D NumPy arrays of shape `(N, 9, 7, 7)` representing a $35\text{ km} \times 35\text{ km}$ spatial grid across 9 standardized environmental channels:
  * *Weather (4):* `t2m`, `rh2m`, `wind_speed_10m`, `tp`
  * *Vegetation (2):* `ndvi`, `evi`
  * *Topography (3):* `elevation`, `slope`, `aspect`
* **Data Splits & Sampling:**
  * **2022 Training:** 557,616 samples (100% fire ignitions, deterministic 2% background negative sample).
  * **2023 Validation:** 1,388,899 samples (100% fire ignitions, deterministic 5% background negative sample).
* **Architecture (`WildfireCNN` in `src/models.py`):**
  * `Conv2D(9 -> 32, 3x3)` $\to$ `BatchNorm2D` $\to$ `ReLU`
  * `Conv2D(32 -> 64, 3x3)` $\to$ `BatchNorm2D` $\to$ `ReLU`
  * `AdaptiveAvgPool2D((1, 1))` $\to$ `Flatten`
  * `Linear(64 -> 32)` $\to$ `ReLU` $\to$ `Dropout(p=0.3)` $\to$ `Linear(32 -> 1)` (raw logits)
* **Optimization & Training Strategy:**
  * **Loss:** `nn.BCEWithLogitsLoss(pos_weight=50.0)` to handle extreme class imbalance.
  * **Optimizer & Schedule:** `AdamW(lr=1e-3, weight_decay=1e-4)` with `ReduceLROnPlateau(mode='max', factor=0.5, patience=1)` stepping on Validation PR-AUC.
  * **Hardware Pipeline:** Automatic device fallback (`CUDA` $\to$ `MPS` $\to$ `CPU`).
* **Experimental Benchmarks:**
  * **Run 1 (Unseeded):** Peak Validation PR-AUC = `0.9317`, ROC-AUC = `0.9989`.
  * **Run 2 (Seeded - `set_seed(42)`):** Peak Validation PR-AUC = `0.9214`, ROC-AUC = `0.9988`, Final Validation Loss = `0.0232`.
  * **Artifacts:** Checkpoint saved to `models/cnn_baseline_run2_seeded.pt`, diagnostics saved to `outputs/cnn_baseline_run2_seeded_diagnostics.png`, and metrics logged to `outputs/cnn_baseline_run2_seeded_validation_metrics.json`.

---

### 2. `model_experiments_logreg.ipynb` — Tabular Logistic Regression Baseline
* **Goal:** Establish a baseline using a linear model trained on single-pixel tabular feature vectors without spatial context.
* **Pipeline:** DuckDB temporal join extracting day $t-1$ weather against day $t$ fire occurrences, `SimpleImputer(strategy='median')`, `StandardScaler()`, and `LogisticRegression(max_iter=1000, random_state=42)` weighted $50\times$ on the background class[cite: 3].
* **2023 Validation Performance:**
  * **PR-AUC:** `0.0105` (No-skill baseline: `0.00013`)[cite: 3]
  * **ROC-AUC:** `0.9646`[cite: 3]
  * **Optimal Threshold (0.002):** Precision = `0.0099`, Recall = `0.6663`, F1 = `0.0196`[cite: 3]
* **Key Finding:** While linear separating hyperplanes achieve high ROC-AUC due to dominant true negatives, the lack of spatial interaction causes heavy false positive generation and poor precision on rare fire events[cite: 3].

---

### 3. `model_experiments_random_forest.ipynb` — Tabular Random Forest
* **Goal:** Non-linear decision tree ensemble modeling on 9 environmental features.
* **Hyperparameters:** `n_estimators=100`, `max_depth=12`, `min_samples_leaf=10`, `max_features='sqrt'`, `random_state=42`, `n_jobs=4`, with positive fire sample weight = 1.0 and negative background weight = 50.0[cite: 10].
* **2023 Validation Performance:**
  * **PR-AUC:** `0.4493`[cite: 10]
  * **ROC-AUC:** `0.9922`[cite: 10]
  * **Threshold Analysis (Best F1 = 0.1819):** Precision = `0.4073`, Recall = `0.5499`, F1 = `0.4680`[cite: 10]
* **Feature Importance:** `elevation` (27.2%), `ndvi` (13.8%), `evi` (13.8%), `slope` (12.3%), `aspect` (11.3%), `t2m` (8.9%), `rh2m` (7.5%), `wind_speed_10m` (4.7%), `tp` (0.7%)[cite: 10].

---

### 4. `model_experiments_xgboost.ipynb` — Gradient Boosted Decision Trees
* **Goal:** High-capacity gradient boosting on 9 tabular environmental features.
* **Hyperparameters:** `n_estimators=300`, `max_depth=5`, `learning_rate=0.05`, `min_child_weight=5`, `subsample=1.0`, `colsample_bytree=1.0`, `tree_method='hist'`, `eval_metric='logloss'`, `random_state=42`, sample weights adjusting for the 2% negative subsampling[cite: 11].
* **2023 Validation Performance:**
  * **PR-AUC:** `0.2673`[cite: 11]
  * **ROC-AUC:** `0.9482`[cite: 11]
  * **Threshold Analysis (Best F1 = 0.6900):** Precision = `0.3088`, Recall = `0.5641`, F1 = `0.3991`[cite: 11]
* **Feature Importance:** Dominated by vegetation and terrain: `evi` (71.0%), `elevation` (21.4%), `ndvi` (5.0%), `tp` (1.0%), `rh2m` (0.6%), `t2m` (0.4%), `slope` (0.3%), `wind_speed_10m` (0.1%), `aspect` (0.1%)[cite: 11].

---

### 5. `model_experiments_ablation.ipynb` — Feature Group Ablation Study
* **Goal:** Quantify the incremental predictive value of environmental subsystems by training separate Random Forest models across feature subsets[cite: 9]:
  1. **Weather Only (4 features):** `t2m`, `rh2m`, `wind_speed_10m`, `tp`[cite: 9]
  2. **Weather + Vegetation (6 features):** Weather features + `ndvi`, `evi`[cite: 9]
  3. **Full Model (9 features):** Weather + Vegetation + `elevation`, `slope`, `aspect`[cite: 9]
* **Ablation Trajectory Across Splits:**
  * **2023 Validation PR-AUC:** Weather Only (`0.0003`) $\to$ Weather + Veg (`0.0059`) $\to$ Full Model (`0.4493`)[cite: 9]
  * **2024 Test PR-AUC:** Weather Only (`0.0006`) $\to$ Weather + Veg (`0.0103`) $\to$ Full Model (`0.4458`)[cite: 9]
* **Takeaway:** Atmospheric variables alone have negligible predictive capability for ignition localization; adding vegetation indices provides an initial order-of-magnitude boost, but static topographical features (elevation, slope, aspect) are responsible for the vast majority of tabular predictive performance[cite: 9].

---

### 6. `model_experiments_2024_test.ipynb` — 2024 Independent Holdout Evaluation
* **Goal:** Execute a batch evaluation of all trained tabular models against the completely un-sampled, full-population 2024 test partition[cite: 6].
* **Test Population:** Streamed via DuckDB Arrow readers across **27,731,454 observations** (3,843 confirmed fire ignitions; positive prevalence $\approx 0.0138\%$)[cite: 6].
* **Model Comparison Table (`outputs/model_comparison_2024.csv`):**

| Model | Evaluated Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 Score | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression**[cite: 6] | 0.0020[cite: 6] | 0.0120[cite: 6] | 0.9657[cite: 6] | 0.0112[cite: 6] | 0.6419[cite: 6] | 0.0220[cite: 6] | 218,140[cite: 6] |
| **XGBoost**[cite: 6] | 0.6900[cite: 6] | 0.2704[cite: 6] | 0.9458[cite: 6] | 0.3329[cite: 6] | 0.5587[cite: 6] | 0.4172[cite: 6] | 4,302[cite: 6] |
| **Random Forest**[cite: 6] | 0.1819[cite: 6] | **0.4458**[cite: 6] | **0.9910**[cite: 6] | **0.4189**[cite: 6] | 0.5438[cite: 6] | **0.4733**[cite: 6] | **2,899**[cite: 6] |

---

## Environment & Dependency Requirements

The notebooks in this repository are verified on **Python 3.12.5 / 3.13.2** with the following pinned core libraries[cite: 3, 10, 11]:
* `torch` $\ge$ 2.2.0 (with CUDA or Apple Silicon MPS support)
* `scikit-learn`: `1.9.1`[cite: 10, 11]
* `duckdb`: `2.0.0`[cite: 10, 11]
* `xgboost`: `3.4.1`[cite: 11]
* `pyarrow`: `26.0.0`[cite: 10, 11]
* `pandas`: `3.0.6`[cite: 10, 11]
* `numpy`: `2.5.4`[cite: 10, 11]
* `joblib`: `1.6.0`[cite: 11]