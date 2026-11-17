# Apples-to-Apples Model Comparison Report

**Cohort:** 118 Subjects (4 Institutions: Peking N=69, KKI N=24, NeuroIMAGE N=19, NYU N=6)  
**Sample Balance:** 68 Control (57.63%), 50 ADHD (42.37%)  
**Evaluator Status:** Freshly refitted inside each training fold (no pre-fitted model or scaler leaks)  

---

## 1. Executive Summary Table

| Metric / Dimension | Majority Baseline | Age & Sex Only | Locked Production Pipeline (Random Forest) | Final Candidate Pipeline (Nested Ensemble) | Superior Model / Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pipeline Architecture** | Naive Majority | `StandardScaler -> LogisticRegression(C=1.0, balanced)` | `MedianImputer -> StandardScaler -> SelectKBest(k=100) -> RF(50, d=3, balanced)` | Nested Grid Selection over 126 configs (Pearson/Partial/Tangent + K=50/100/200 + LogReg/SVM/RF/Ensemble) | — |
| **LOIO ROC-AUC** | 0.5000 | **0.5857** [0.489, 0.685] | **0.6250** [0.518, 0.726] | **0.6253** [0.521, 0.728] | Neither connectivity model significantly beats the other |
| **LOIO Accuracy** | 57.63% | **61.02%** [52.5%, 69.5%] | **61.86%** [53.4%, 70.3%] | **58.47%** [48.3%, 67.8%] | Comparable (~58.5% - 61.0%) |
| **LOIO Balanced Acc** | 50.00% | **63.26%** [55.1%, 71.0%] | **60.82%** [51.8%, 69.9%] | **56.03%** [46.8%, 64.7%] | Demographics higher sensitivity |
| **LOIO Sensitivity** | 0.00% | **78.00%** [66.0%, 88.9%] | **54.00%** [40.5%, 67.9%] | **40.00%** [26.3%, 53.3%] | Demographics: 78.0%, Final: 40.0% |
| **LOIO Specificity** | 100.00% | **48.53%** [36.8%, 60.6%] | **67.65%** [56.1%, 78.1%] | **72.06%** [60.3%, 82.5%] | Locked RF / Final Cand: ~72% |
| **Stratified 5-Fold AUC** | 0.5000 | — | **0.6148 ± 0.0171** | **0.5831 ± 0.0400** | Repeated 5x5 CV |
| **Permutation p-value** | 1.0000 | p = **0.0498** (null: 0.4832) | p = **0.0100** (null: 0.4851) | p = **0.0050** (null: 0.4881) | 200 within-institution permutations |

---

## 2. Statistical Significance & Paired Hypothesis Tests

### Test 1: Final Candidate vs. Locked Random Forest
* **Hypothesis:** Does the complex nested ensemble outperform the frozen, parsimonious Balanced Random Forest?
* $\Delta\text{AUC} = \text{AUC}_{\text{Candidate}} - \text{AUC}_{\text{Locked RF}}$:
  $$\Delta\text{AUC} = +0.0003 \quad [95\% \text{ CI: } -0.0633, +0.0639]$$
* **One-sided p-value:** $p = 0.4960$
* **Conclusion:** **Non-significant.** The confidence interval comfortably crosses zero. The 126-configuration nested ensemble search yields no statistically demonstrable advantage over the simple frozen Balanced Random Forest on this sample.

### Test 2: Locked Random Forest vs. Age/Sex Demographics
* **Hypothesis:** Does functional connectivity alone (using the locked RF) outperform non-imaging demographic covariates?
* $\Delta\text{AUC} = \text{AUC}_{\text{Locked RF}} - \text{AUC}_{\text{Age/Sex}}$:
  $$\Delta\text{AUC} = +0.0397 \quad [95\% \text{ CI: } -0.1061, +0.1695]$$
* **One-sided p-value:** $p = 0.2670$
* **Conclusion:** **Non-significant.** Functional connectivity does not statistically outperform demographic covariates alone.

### Test 3: Final Candidate vs. Age/Sex Demographics
* $\Delta\text{AUC} = +0.0400 \quad [95\% \text{ CI: } -0.1038, +0.1714]$
* **One-sided p-value:** $p = 0.2685$
* **Conclusion:** **Non-significant.**

---

## 3. Scientific Assessment & Recommendation

1. **Neither Pipeline Demonstrates Superiority:**
   Under identical 4-fold Leave-One-Institution-Out CV, both the locked production pipeline (Balanced RF 50 trees, depth 3) and the 126-configuration nested candidate ensemble produce comparable generalization performance:
   * Locked Balanced RF: ROC-AUC = **0.6250** [0.518, 0.726], Accuracy = **61.86%**
   * Final Candidate Ensemble: ROC-AUC = **0.6253** [0.521, 0.728], Accuracy = **58.47%**
   * Paired difference: $\Delta\text{AUC} = +0.0003$, $p = 0.4960$.

2. **Principle of Parsimony:**
   The locked Balanced Random Forest has clear engineering and interpretability advantages:
   * Single, fixed estimator with 50 shallow trees (depth 3).
   * Fast, deterministic, and easily auditable via TreeSHAP.
   * Does not require computing Riemannian tangent-space projections or tuning multi-model meta-learners.

3. **Dashboard Integrity:**
   As instructed, the production dashboard model remains the locked Balanced Random Forest. The candidate study results serve as an exhaustive methodological benchmark confirming the limits of resting-state fMRI classification on this cohort.
