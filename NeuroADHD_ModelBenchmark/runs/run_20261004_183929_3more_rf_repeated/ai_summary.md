# Balanced Random Forest Repeated Cross-Validation Robustness Report

**Run ID:** `run_20261004_183929_3more_rf_repeated`  
**Experiment Type:** 3 Additional Repetitions (Seeds: 47, 48, 49) + Full 8-Repetition Analysis (Seeds: 42–49)  
**Dataset:** 133 Development Subjects (93 Control, 40 ADHD across 7 sites)  
**Peking_2 Test Status:** **STRICTLY LOCKED & UNTOUCHED (N=20)**  

---

## 1. Executive Summary

### A. New 3 Repetitions (Seeds: 47, 48, 49)
- **Mean ROC-AUC:** `0.6022 ± 0.0085` (Min: `0.5927`, Max: `0.6091`)
- **Mean Balanced Accuracy:** `0.6078 ± 0.0144` (Min: `0.5941`, Max: `0.6227`)
- **Mean Sensitivity (ADHD Recall):** `0.5167 ± 0.0144` (Min: `0.5000`, Max: `0.5250`)
- **Mean F1 Score:** `0.4663 ± 0.0167` (Min: `0.4494`, Max: `0.4828`)
- **Mean Accuracy:** `0.6441 ± 0.0157`

### B. Full 8 Repetitions (Seeds: 42, 43, 44, 45, 46, 47, 48, 49)
- **Mean ROC-AUC ± SD:** `0.6045 ± 0.0163` (Min: `0.5892`, Max: `0.6403`)
- **Mean PR-AUC ± SD:** `0.3815 ± 0.0224`
- **Mean Accuracy ± SD:** `0.6447 ± 0.0138`
- **Mean Balanced Accuracy ± SD:** `0.6026 ± 0.0163` (Min: `0.5781`, Max: `0.6281`)
- **Mean Sensitivity ± SD:** `0.4969 ± 0.0339`
- **Mean Specificity ± SD:** `0.7083 ± 0.0194`
- **Mean F1 ± SD:** `0.4566 ± 0.0232`

---

## 2. Comparison under Identical Repeated-CV Protocol

| Model / Architecture | Protocol | ROC-AUC (Mean ± SD) | Balanced Accuracy (Mean ± SD) | Sensitivity (Mean ± SD) | F1 Score (Mean ± SD) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Balanced Random Forest** | **8-Rep Repeated CV** | **0.6045 ± 0.0163** | **0.6026 ± 0.0163** | **0.4969 ± 0.0339** | **0.4566 ± 0.0232** |
| **Balanced Random Forest** | **New 3-Rep CV (47-49)** | **0.6022 ± 0.0085** | **0.6078 ± 0.0144** | **0.5167 ± 0.0144** | **0.4663 ± 0.0167** |
| Soft Voting (RF + XGB) | 5-Rep Repeated CV | 0.5926 ± 0.0229 | 0.5798 ± 0.0226 | 0.4500 ± 0.0354 | 0.4234 ± 0.0299 |
| Hard Voting (RF + XGB) | 5-Rep Repeated CV | 0.5926 ± 0.0229 | 0.5798 ± 0.0226 | 0.4500 ± 0.0354 | 0.4234 ± 0.0299 |
| Stacking (Balanced LogReg) | 5-Rep Repeated CV | 0.5555 ± 0.0280 | 0.5428 ± 0.0191 | 0.5050 ± 0.0485 | 0.4065 ± 0.0263 |

---

## 3. Critical Protocol Distinction

- **Historical Single Validation (NeuroIMAGE split, N=19):** ROC-AUC = 0.7000, BalAcc = 0.6833, Sensitivity = 0.7000, F1 = 0.7000.
- **Repeated 5-Fold Cross-Validation (Full 133 subjects, 8 independent splits):** ROC-AUC = 0.6045 ± 0.0163, BalAcc = 0.6026 ± 0.0163.
- The single partition score reflects a favorable holdout site, whereas the 8-repetition CV provides an unbiased generalization estimate across all sites.
