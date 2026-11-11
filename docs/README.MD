# Project Documentation (`docs/`)

This directory houses high-level project documentation, formal deliverable reports, academic slide presentations, technical engineering breakdowns, and visual asset collateral for the California Wildfire Prediction project.

---

## Directory Inventory

```
docs/
├── assets/                                      # Rendered figures, architectural diagrams, and report graphics
├── Group_24_Project_Proposal_COSC_4337.pdf      # Initial formal project specification and research proposal
├── Group_24_Wildfire_Prediction_Presentation.pdf# Slide deck covering methodology, classical baselines, and findings
├── README.md                                    # Directory documentation catalog (this file)
├── transform_master_dataset_breakdown.md        # Technical specification for 4D spatial tensor extraction
└── transform_master_dataset_breakdown.rtf       # Formatted rich text export of the transformation breakdown
```

---

## Document Details

### 1. `transform_master_dataset_breakdown.md`
A technical data-engineering specification detailing the transformation of tabular observations from `data/processed/master_training_dataset.parquet` into 4D spatial tensors for convolutional deep learning:
* **Spatial Framing:** Explains the logic behind extracting $7 \times 7$ grid patches ($35\text{ km} \times 35\text{ km}$ spatial receptive field) centered on target grid cells.
* **Multi-Modal Channel Ordering:** Defines the exact 9-channel stacking convention (4 weather, 2 vegetation, 3 topography).
* **Temporal Lag & Padding:** Details the prior-day ($t-1$) feature alignment relative to day $t$ ignition outcomes, boundary reflection padding, and edge-handling rules.
* **Partition Splits & Sampling:** Catalogs the deterministic negative subsampling rates for the 2022 Train (2%), 2023 Validation (5%), and 2024 Test (5%) `.npy` tensor partitions.

### 2. Formal Academic Deliverables
* **`Group_24_Project_Proposal_COSC_4337.pdf`**: The original foundational proposal outlining problem motivation, dataset acquisition pipelines (ERA5, MODIS, USGS 3DEP, FPA-FOD), evaluation metric selection (PR-AUC vs. ROC-AUC), and the initial research roadmap.
* **`Group_24_Wildfire_Prediction_Presentation.pdf`**: Comprehensive team presentation summarizing:
  * The extreme class imbalance problem in California wildfire prediction ($\sim 0.013\%$ fire prevalence).
  * Data consolidation and deduplication across 83.4M grid-day observations.
  * Classical model baselines (Logistic Regression, Random Forest, XGBoost) and feature importance rankings.
  * The ablation study proving the dominant predictive power of topography and vegetation over weather alone.

### 3. `docs/assets/`
Storage for raw visual assets, pipeline flowcharts, GIS maps, PR/ROC evaluation curves, and confusion matrix plots embedded throughout project reports and presentation decks.