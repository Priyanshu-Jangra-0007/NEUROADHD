# NeuroADHD Hybrid & Ensemble Experiment Report — run_20261004_182834_ensemble

**Generated:** 2026-10-04 18:28:44 UTC  
**Status:** Completed  
**Test Set Status:** 🔒 **LOCKED & UNTOUCHED** (Peking_2 site, 20 subjects)

---

## 1. Objective & Methodology
This experiment systematically tests whether combining our strongest individual models improves out-of-site generalization over our current best single model (**Balanced Random Forest**).

### Base Models Evaluated ($K=100$, Preprocessing fit strictly on training set):
1. **Random Forest**: `n_estimators=50, max_depth=3, class_weight='balanced'`
2. **XGBoost**: `n_estimators=100, max_depth=2, learning_rate=0.05, scale_pos_weight=2.8`
3. **Extra Trees**: `n_estimators=50, max_depth=5, class_weight='balanced'`
4. **RBF SVM**: `C=1.0, gamma='scale', class_weight='balanced'`
5. **Logistic Regression**: `C=0.1, penalty='l1', solver='saga', class_weight='balanced'`

---

## 2. Category Winner Ensembles

### A. Best Hard Voting Ensemble
- **Model Combination:** `Hard Voting (RF + XGBoost)`
- **Validation ROC-AUC:** `0.6778`
- **Validation Balanced Accuracy:** `0.6889`
- **Validation Sensitivity:** `0.6000`
- **Validation F1 Score:** `0.6667`

### B. Best Soft Voting Ensemble
- **Model Combination:** `Soft Voting Weighted (RF + XGBoost)`
- **Validation ROC-AUC:** `0.6889`
- **Validation Balanced Accuracy:** `0.6889`
- **Validation Sensitivity:** `0.6000`
- **Validation F1 Score:** `0.6667`

### C. Best Stacking Ensemble
- **Model Combination:** `Stacking (Logistic Regression Meta on RF + XGBoost + Extra Trees + Logistic Regression)`
- **Validation ROC-AUC:** `0.6889`
- **Validation Balanced Accuracy:** `0.5000`
- **Validation Sensitivity:** `0.0000`
- **Validation F1 Score:** `0.0000`

---

## 3. Best Overall Ensemble vs. Single Model Benchmark

| Metric | Balanced Random Forest (Single Best) | Best Ensemble (Soft Voting Weighted (RF + XGBoost)) | Delta (Ensemble vs Single) |
| :--- | :---: | :---: | :---: |
| **Validation ROC-AUC** | `0.7000` | **`0.6889`** | `-0.0111` |
| **Validation PR-AUC** | `0.5959` | **`0.5911`** | `-0.0048` |
| **Validation Balanced Accuracy** | `0.6833` | **`0.6889`** | `+0.0056` |
| **Validation Sensitivity (ADHD)** | `0.7000` | **`0.6000`** | `-0.1000` |
| **Validation Specificity (Control)** | `0.6667` | **`0.7778`** | `+0.1111` |
| **Validation F1 Score** | `0.7000` | **`0.6667`** | `-0.0333` |
| **Validation Accuracy** | `0.6842` | **`0.6842`** | `+0.0000` |

---

## 4. Multi-Seed Stability Summary (Seeds 42, 43, 44, 45, 46)

### Rank #1: Soft Voting Weighted (RF + XGBoost)
- **Val ROC-AUC:** 0.6622 ± 0.0151 (Range: 0.6444 – 0.6889)
- **Val Balanced Acc:** 0.6889 ± 0.0000
- **Val Sensitivity (ADHD):** 0.6000 ± 0.0000
- **Val F1 Score:** 0.6667 ± 0.0000

### Rank #2: Stacking (Logistic Regression Meta on RF + XGBoost + Extra Trees + Logistic Regression)
- **Val ROC-AUC:** 0.6578 ± 0.0163 (Range: 0.6444 – 0.6889)
- **Val Balanced Acc:** 0.5000 ± 0.0000
- **Val Sensitivity (ADHD):** 0.0000 ± 0.0000
- **Val F1 Score:** 0.0000 ± 0.0000


---

## 5. Test Set Evaluation Declaration

> **TEST EVALUATION:**  
> **FINAL TEST NOT PERFORMED — Peking_2 remains locked.**
