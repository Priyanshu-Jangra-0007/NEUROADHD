# NeuroADHD Project Cleanup & Integrity Report

**Date:** 2026-10-05  
**Scope:** Repository Cleanup, Artifact Preservation, and Reproducibility Integrity Check  

---

## 1. Summary of Cleanup Actions

- **Operating System Metadata Cleaned:** 8 `.DS_Store` files removed across non-venv directories.
- **Python Bytecode Cache Cleaned:** Non-venv `__pycache__` directories removed (auto-regenerating).
- **Empty Work Directories Removed:** 1 empty temporary folder (`work/`) removed.
- **Archive Directory Established:** `_archive_cleanup/` created at workspace root for non-destructive quarantine if needed.
- **Scientific Data & Artifacts Protected:** 100% of raw/processed ADHD-200 fMRI datasets, split manifests, trained model binaries, explainability matrices, and repeated cross-validation logs preserved untouched.

---

## 2. Integrity Verification of 16 Core Artifacts

| # | Required Artifact | Path | Status |
| :---: | :--- | :--- | :---: |
| 1 | `final_model.joblib` | `NeuroADHD_ModelBenchmark/runs/run_20261004_182318_final_locked/final_model.joblib` | **VERIFIED** |
| 2 | `final_preprocessing_pipeline.joblib` | `NeuroADHD_ModelBenchmark/runs/run_20261004_182318_final_locked/final_preprocessing_pipeline.joblib` | **VERIFIED** |
| 3 | `final_test_results.json` | `NeuroADHD_ModelBenchmark/runs/run_20261004_182318_final_locked/final_test_results.json` | **VERIFIED** |
| 4 | `final_test_predictions.csv` | `NeuroADHD_ModelBenchmark/runs/run_20261004_182318_final_locked/final_test_predictions.csv` | **VERIFIED** |
| 5 | `shap_global_importance.csv` | `NeuroADHD_ModelBenchmark/runs/run_20261004_184658_explainability/explainability/shap_global_importance.csv` | **VERIFIED** |
| 6 | `shap_subject_values.csv` | `NeuroADHD_ModelBenchmark/runs/run_20261004_184658_explainability/explainability/shap_subject_values.csv` | **VERIFIED** |
| 7 | `permutation_importance.csv` | `NeuroADHD_ModelBenchmark/runs/run_20261004_184658_explainability/explainability/permutation_importance.csv` | **VERIFIED** |
| 8 | `top_connectivity_edges.csv` | `NeuroADHD_ModelBenchmark/runs/run_20261004_184658_explainability/explainability/top_connectivity_edges.csv` | **VERIFIED** |
| 9 | `roi_importance_summary.csv` | `NeuroADHD_ModelBenchmark/runs/run_20261004_184658_explainability/explainability/roi_importance_summary.csv` | **VERIFIED** |
| 10 | `cc200_roi_mapping.csv` | `NeuroADHD_ModelBenchmark/runs/run_20261004_185219_atlas_mapping/atlas_mapping/cc200_roi_mapping.csv` | **VERIFIED** |
| 11 | `mapped_connectivity_edges.csv` | `NeuroADHD_ModelBenchmark/runs/run_20261004_185219_atlas_mapping/atlas_mapping/mapped_connectivity_edges.csv` | **VERIFIED** |
| 12 | `mapped_hub_rois.csv` | `NeuroADHD_ModelBenchmark/runs/run_20261004_185219_atlas_mapping/atlas_mapping/mapped_hub_rois.csv` | **VERIFIED** |
| 13 | `network_connectivity_summary.csv` | `NeuroADHD_ModelBenchmark/runs/run_20261004_185219_atlas_mapping/atlas_mapping/network_connectivity_summary.csv` | **VERIFIED** |
| 14 | Repeated RF Robustness Statistics | `NeuroADHD_ModelBenchmark/runs/run_20261004_183929_3more_rf_repeated/rf_summary_statistics.json` | **VERIFIED** |
| 15 | Repeated Hybrid Robustness Statistics | `NeuroADHD_ModelBenchmark/runs/run_20261004_183509_repeated_hybrid_01/summary_statistics.json` | **VERIFIED** |
| 16 | Subject Split Manifests | `NeuroADHD/data/adhd200/processed/splits/subject_split_manifest.csv` | **VERIFIED** |

---

## 3. Scientific Immutability Checklist

- [x] Preprocessing definitions (`SimpleImputer` $\rightarrow$ `VarianceThreshold` $\rightarrow$ `StandardScaler` $\rightarrow$ `SelectKBest(k=100)`) unchanged.
- [x] Balanced Random Forest hyperparameters ($K=100$, $N=50$, $\text{depth}=3$, $\text{min\_split}=2$, $\text{class\_weight}=\text{balanced}$) unchanged.
- [x] Peking_2 test split remains strictly locked with original final test evaluation recorded.
- [x] No data fabrication or artificial anatomical labels introduced.
