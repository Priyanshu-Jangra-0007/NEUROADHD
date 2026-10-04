# NeuroADHD Model Optimization Report — run_20261004_180123_opt1

**Generated:** 2026-10-04 18:05:51 UTC  
**Status:** Completed  
**Test Set Status:** 🔒 **LOCKED & UNTOUCHED** (Peking_2 site, 20 subjects: 10 Control, 10 ADHD)

---

## 1. Verified Authoritative Splits
- **Training Set:** 114 subjects (84 Control, 30 ADHD) across 6 sites
- **Validation Set:** 19 subjects (9 Control, 10 ADHD) — NeuroIMAGE site
- **Test Set:** 20 subjects (10 Control, 10 ADHD) — Peking_2 site (UNTOUCHED)

---

## 2. Key Optimization Results & Class Balancing Impact

### Baseline Reference (Run 1)
- Model: Random Forest ($k=100$, Unweighted)
- Val ROC-AUC: `0.7333` | Val Acc: `0.4211` | Val Sens (ADHD): `0.1000` | Val F1: `0.1538`

### Best New Configuration Found
- **Model:** `random_forest`
- **Feature Count ($k$):** `100`
- **Class Weighting:** `unweighted`
- **Best Hyperparameters:** `{"class_weight": null, "max_depth": 3, "max_features": "sqrt", "n_estimators": 50, "random_state": 42}`

#### Validation Metrics Comparison:
- **Val ROC-AUC:** `0.7222` (Delta: `-0.0111`)
- **Val Balanced Accuracy:** `0.4389` (Delta: `-0.0000`)
- **Val ADHD Sensitivity (Recall):** `0.1000` (Delta: `+0.0000`)
- **Val F1 Score:** `0.1538` (Delta: `+0.0000`)
- **Val Specificity (Control):** `0.7778`

---

## 3. Multi-Seed Stability Summary (5 Seeds: 42, 43, 44, 45, 46)

### random_forest (k=100, CW=unweighted)
- **Val ROC-AUC:** 0.6733 ± 0.0295 (Range: 0.6333 – 0.7222)
- **Val Balanced Acc:** 0.4289 ± 0.0200
- **Val Sensitivity (ADHD):** 0.0800 ± 0.0400
- **Val F1 Score:** 0.1231 ± 0.0615

### random_forest (k=100, CW=balanced)
- **Val ROC-AUC:** 0.6689 ± 0.0215 (Range: 0.6444 – 0.7000)
- **Val Balanced Acc:** 0.6833 ± 0.0351
- **Val Sensitivity (ADHD):** 0.7000 ± 0.0000
- **Val F1 Score:** 0.7007 ± 0.0222

### logistic_regression (k=200, CW=unweighted)
- **Val ROC-AUC:** 0.7000 ± 0.0000 (Range: 0.7000 – 0.7000)
- **Val Balanced Acc:** 0.4444 ± 0.0000
- **Val Sensitivity (ADHD):** 0.0000 ± 0.0000
- **Val F1 Score:** 0.0000 ± 0.0000


---

## 4. Test Set Evaluation Declaration

> **IMPORTANT DECLARATION:**  
> **Final test evaluation not performed during this optimization run.**  
> The Peking_2 test set remained completely locked and untouched throughout model tuning, feature selection, and class-weight evaluation.
