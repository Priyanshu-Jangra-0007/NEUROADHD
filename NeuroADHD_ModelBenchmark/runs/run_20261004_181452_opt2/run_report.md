# NeuroADHD Focused Model Optimization Report (Opt-2) — run_20261004_181452_opt2

**Generated:** 2026-10-04 18:20:59 UTC  
**Status:** Completed  
**Test Set Status:** 🔒 **LOCKED & UNTOUCHED** (Peking_2 site, 20 subjects)

---

## 1. Candidate Scope & Exact Opt-1 Match Verification
- **Exact Opt-1 Match:** Random Forest ($K=100$, `class_weight="balanced"`) & Extra Trees ($K=50$, `class_weight="balanced"`) produced Validation Accuracy = 0.6842, Balanced Accuracy = 0.6833, Sensitivity = 0.7000, F1 = 0.7000.
- **Candidates Evaluated in Opt-2:**
  1. Random Forest + $K=50$ + `class_weight="balanced"`
  2. Random Forest + $K=100$ + `class_weight="balanced"`
  3. Extra Trees + $K=50$ + `class_weight="balanced"`
  4. Extra Trees + $K=100$ + `class_weight="balanced"`

---

## 2. Best Final Candidate Selected
- **Model:** `Random Forest`
- **Features Selected ($K$):** `100`
- **Class Weight:** `balanced_subsample`
- **Best Hyperparameters:** `{"class_weight": "balanced_subsample", "max_depth": null, "max_features": "sqrt", "min_samples_leaf": 1, "min_samples_split": 2, "n_estimators": 50, "random_state": 42}`

### Validation Performance (NeuroIMAGE Holdout — 19 Subjects)
- **Validation ROC-AUC:** `0.7167`
- **Validation PR-AUC:** `0.6097`
- **Validation Balanced Accuracy:** `0.5444`
- **Validation Sensitivity (ADHD Recall):** `0.2000`
- **Validation Specificity (Control Recall):** `0.8889`
- **Validation F1 Score:** `0.3077`
- **Validation Accuracy:** `0.5263`
- **Training CV ROC-AUC (5-Fold Mean ± Std):** `0.6102 ± 0.1370`
- **Multi-Seed Val ROC-AUC (5 Seeds Mean ± Std):** `0.6967 ± 0.0269`

---

## 3. Three-Way Performance Comparison

| Paradigm | Model & Config | Val ROC-AUC | Val PR-AUC | Val Bal. Acc | Val Sensitivity | Val Specificity | Val F1 | Val Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Baseline** | Random Forest ($K=100$, Unweighted) | **0.7333** | **0.7516** | 0.4389 | 0.1000 | **0.7778** | 0.1538 | 0.4211 |
| **Opt-1 Optimization** | Random Forest ($K=100$, Balanced) | 0.7000 | 0.5959 | **0.6833** | **0.7000** | 0.6667 | **0.7000** | **0.6842** |
| **Opt-2 Focused Opt** | Random Forest ($K=100$, Balanced) | **0.7167** | **0.6097** | **0.5444** | **0.2000** | 0.8889 | **0.3077** | **0.5263** |

---

## 4. Test Set Evaluation Declaration

> **IMPORTANT DECLARATION:**  
> **Final test evaluation not performed during this optimization run.**  
> The Peking_2 test set remained completely locked and untouched.
