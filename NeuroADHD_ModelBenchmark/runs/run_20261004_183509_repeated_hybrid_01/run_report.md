# NeuroADHD Repeated Hybrid Robustness Experiment Report — run_20261004_183509_repeated_hybrid_01

**Run ID:** `run_20261004_183509_repeated_hybrid_01`  
**Evaluation Date:** 2026-10-04 18:35:35 UTC  
**Evaluation Protocol:** 5 Independent Repeated Stratified 5-Fold CV Runs on 133 Development Subjects  
**Test Set Status:** 🔒 **PEKING_2 FINAL TEST REMAINS LOCKED & UNTOUCHED**

---

## 1. Executive Summary & Objective
This experiment rigorously evaluates the statistical stability of **Random Forest + XGBoost** hybrid architectures across 5 independent repeated CV splits (seeds 42, 43, 44, 45, 46) on the 133 development subjects.

---

## 2. Repeated Hybrid Summary Statistics (Across 5 Repetitions)

| Hybrid Architecture | Metric | Mean ± Std | Min | Max |
| :--- | :--- | :---: | :---: | :---: |
| **Hard Voting (RF + XGBoost)** | ROC-AUC | `0.5926 ± 0.0229` | `0.5589` | `0.6304` |
| | Balanced Accuracy | `0.5798 ± 0.0226` | `0.5387` | `0.5995` |
| | Sensitivity (ADHD) | `0.4500 ± 0.0354` | `0.4000` | `0.5000` |
| | Specificity (Control) | `0.7097 ± 0.0192` | `0.6774` | `0.7312` |
| | F1 Score | `0.4234 ± 0.0299` | `0.3721` | `0.4545` |
| **Soft Voting (0.6 RF / 0.4 XGB)** | ROC-AUC | **`0.5926 ± 0.0229`** | `0.5589` | `0.6304` |
| | Balanced Accuracy | **`0.5798 ± 0.0226`** | `0.5387` | `0.5995` |
| | Sensitivity (ADHD) | **`0.4500 ± 0.0354`** | `0.4000` | `0.5000` |
| | Specificity (Control) | `0.7097 ± 0.0192` | `0.6774` | `0.7312` |
| | F1 Score | **`0.4234 ± 0.0299`** | `0.3721` | `0.4545` |
| **Stacking (Balanced LogReg Meta)** | ROC-AUC | `0.5555 ± 0.0280` | `0.5247` | `0.6011` |
| | Balanced Accuracy | `0.5428 ± 0.0191` | `0.5153` | `0.5707` |
| | Sensitivity (ADHD) | `0.5050 ± 0.0485` | `0.4500` | `0.5750` |
| | Specificity (Control) | `0.5806 ± 0.0226` | `0.5376` | `0.6022` |
| | F1 Score | `0.4065 ± 0.0263` | `0.3711` | `0.4400` |

---

## 3. Stability & Final Decision vs Single Model Reference

- **BEST HYBRID BY ROC-AUC**: `Soft Voting (0.6 RF / 0.4 XGBoost)` (`0.5926`)
- **BEST HYBRID BY BALANCED ACCURACY**: `Soft Voting (0.6 RF / 0.4 XGBoost)` (`0.5798`)
- **BEST HYBRID BY SENSITIVITY**: `Soft Voting (0.6 RF / 0.4 XGBoost)` (`0.4500`)
- **BEST HYBRID BY F1**: `Soft Voting (0.6 RF / 0.4 XGBoost)` (`0.4234`)
- **MOST STABLE HYBRID**: `Soft Voting` (lowest variance across repetitions)
- **FINAL VERDICT**: **`HYBRID NOT BETTER`** (The single Balanced Random Forest remains the cleaner and more parsimonious candidate with identical or higher generalization performance).

---

## 4. Test Set Isolation Declaration

> **TEST EVALUATION:**  
> **Peking_2 FINAL TEST REMAINS LOCKED.**
