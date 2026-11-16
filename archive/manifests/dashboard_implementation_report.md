# NeuroADHD Research Dashboard Implementation Report

**Status:** Completed & Validated  
**Build Artifacts:** `NeuroADHD_ModelDashboard/frontend/dist/`  
**Frontend Framework:** React 19 + Vite 8  
**Design Paradigm:** Scientific Research Architecture (No excessive gradients or decorative clutter)  

---

## 1. Dashboard Architecture & View Components

| Section | Component | Description | Data Source |
| :--- | :--- | :--- | :--- |
| **1. Overview** | `OverviewView.jsx` | ADHD-200 cohort metrics (153 subjects, 19,900 edges, K=100 selection, Peking_2 holdout). | `benchmarkData.js` / Manifest |
| **2. Model & Pipeline** | `ModelView.jsx` | Frozen Balanced Random Forest parameters & leakage-safe 8-step preprocessing pipeline. | `final_model_metadata.json` |
| **3. Performance** | `PerformanceView.jsx` | Dual evaluation: 8-repetition CV distribution vs. Locked Peking_2 test results & confusion matrix. | `rf_summary_statistics.json`, `final_test_results.json` |
| **4. Hybrid Comparison** | `HybridComparisonView.jsx` | Comparison of Balanced RF vs. Soft/Hard Voting and Stacking across repeated CV splits. | `summary_statistics.json` |
| **5. Explainability** | `ExplainabilityView.jsx` | TreeSHAP global feature importance, ranking table, and directionality for all 100 features. | `shap_global_importance.csv` |
| **6. Connectivity Network** | `NetworkGraphView.jsx` | Interactive topological graph connecting CC200 Hub nodes (size = centrality, color = +ADHD/-ADHD). | `top_connectivity_edges.csv` |
| **7. Subject Explanations** | `IndividualSubjectView.jsx` | Case-by-case decomposition of positive and negative SHAP edge contributions for Dev & Test subjects. | `subject_explanations.csv` |
| **8. Methodology** | `MethodologyView.jsx` | Split protocols, site isolation rationale, and fold-internal preprocessing guarantees. | Project Specifications |
| **9. Limitations** | `LimitationsView.jsx` | Explicit boundaries of validity, non-causal interpretability notice, and prototype status. | Ethics & Governance Specs |

---

## 2. Integrity & Compliance Verification

- [x] **No Fabricated Labels:** All CC200 ROIs are identified by their verified numeric cluster IDs (`CC200 Node X` / `ROI_XXX`).
- [x] **Unchanged Scientific Metrics:** All metrics (8-Rep ROC-AUC 0.6045 ± 0.0163, Peking_2 ROC-AUC 0.5600, Accuracy 0.6000) match authoritative benchmark runs.
- [x] **Peking_2 Lockdown:** Peking_2 test set remains completely locked and unmutated.
- [x] **Independent Build:** Production bundle compiled cleanly in under 250ms with zero errors.
