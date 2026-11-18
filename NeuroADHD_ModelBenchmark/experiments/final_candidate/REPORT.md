# NeuroADHD: Best Honest Candidate Model Benchmark
## Rigorous Evaluation on Config 2 (118 Subjects, 4 Institutions)

### Executive Summary
- **Cohort**: Config 2 comprising **118 subjects** (Control: 68 [57.6%], ADHD: 50 [42.4%]) across 4 institutions (KKI: 24, NeuroIMAGE: 19, Peking: 69, NYU: 6).
- **Majority-Class Accuracy Baseline**: **57.6%** (assigning all subjects to Control yields 57.6% accuracy, 50.0% balanced accuracy, 0.500 AUC).
- **Candidate Space Explored**: **127 distinct pipeline configurations** evaluated per fold across:
  - Connectivity representations: Pearson $r \to z$, Ledoit-Wolf partial correlation, and tangent-space covariance projection (118/118 raw time series available).
  - In-fold univariate ANOVA feature filtering: $k \in \{50, 100, 200\}$.
  - Covariate integration: with and without demographic covariates (standardized age + sex).
  - Class balancing: `class_weight='balanced'`.
  - Classifiers: L2 Logistic Regression ($C \in \{0.01, 0.1\}$), Linear SVM ($C \in \{0.01, 0.1\}$), RBF SVM ($C = 1.0$), Random Forest (50 trees, max depth 3), Extra Trees (50 trees, max depth 3), and automated soft-voting ensemble of top 3 candidates.
- **Primary Generalization Assessment (LOIO)**: **ROC-AUC = 0.625 [0.521, 0.728]**, Accuracy = 58.5% [48.3%, 67.8%] (vs 57.6% majority baseline), Balanced Accuracy = 56.0% [46.8%, 64.7%].
- **Repeated Stratified 5-Fold (5 Repeats)**: **ROC-AUC = 0.599 [0.496, 0.704]**, Accuracy = 58.5% [50.0%, 66.9%] (vs 57.6% majority baseline), Balanced Accuracy = 56.6% [47.8%, 65.2%].
- **Full-Procedure Permutation Significance (200 within-site shuffles re-running full selection)**: **$p = 0.0050$** (Null mean: 0.488 ± 0.055, 95th percentile: 0.583).
- **Comparison Against Age + Sex Alone**: LOIO ROC-AUC = 0.625 vs Covariates Alone LOIO ROC-AUC = 0.586. Paired bootstrap $\Delta\text{AUC} = +0.0400 [-0.1038, +0.1714]$, $p = 0.2685$.

---

### 1. Data Inventory & Available Modalities
| Property | Detail |
|:---|:---|
| Total Subjects | 118 |
| Control / ADHD | 68 (57.6%) / 50 (42.4%) |
| Majority-Class Baseline | 57.6% |
| Institutions | 4 (KKI: 24, NeuroIMAGE: 19, Peking [1+2+3 merged]: 69, NYU: 6) |
| Raw CC200 Time Series Available | **118 / 118 (100.0%)** verified locally |
| Feature Types Computed | Pearson $r \to z$ (19,900), Ledoit-Wolf Partial Correlation (19,900), Tangent-Space Covariance Projection (19,900) |
| Non-Imaging Covariates | Age (standardized in-fold), Sex (binary 0/1) |
| Leakage Safeguards | Inner 3-fold selection inside training folds only; manifold Fréchet mean fit on training folds only; all scaling and KBest selection within-fold. 4/4 automated leakage unit tests passed. |

---

### 2. Primary Evaluation Results
Both primary validation schemes evaluate generalization without data leakage:

| Metric | LOIO Cross-Validation (4 Groups) | Repeated Stratified 5-Fold (5 Repeats, Mean ± SD) | Majority Baseline |
|:---|:---:|:---:|:---:|
| **ROC-AUC** | **0.625 [0.521, 0.728]** | **0.583 ± 0.040** (Pooled: 0.599 [0.496, 0.704]) | 0.500 |
| **Accuracy** | **58.5% [48.3%, 67.8%]** | **58.1% ± 3.1%** (Pooled: 58.5% [50.0%, 66.9%]) | 57.6% |
| **Balanced Accuracy** | **56.0% [46.8%, 64.7%]** | **56.2% ± 3.5%** (Pooled: 56.6% [47.8%, 65.2%]) | 50.0% |
| **Sensitivity (Recall)** | 40.0% [26.3%, 53.3%] | 43.6% ± 7.2% | 0.0% |
| **Specificity** | 72.1% [60.3%, 82.5%] | 68.8% ± 2.5% | 100.0% |
| **Matthews Corr. Coeff. (MCC)** | 0.127 [-0.068, 0.309] | 0.127 ± 0.070 | 0.000 |

---

### 3. Institution-by-Institution Breakdown (LOIO)
| Institution | N (Ctrl / ADHD) | Accuracy (vs Majority) | Balanced Accuracy | ROC-AUC | Sensitivity | Specificity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **KKI** | 24 (16 / 8) | 54.2% (vs 66.7%) | 53.1% | 0.547 | 50.0% | 56.2% |
| **NYU** | 6 (2 / 4) | 66.7% (vs 66.7%) | 75.0% | 0.625 | 50.0% | 100.0% |
| **NeuroIMAGE** | 19 (9 / 10) | 52.6% (vs 52.6%) | 53.9% | 0.667 | 30.0% | 77.8% |
| **Peking** | 69 (41 / 28) | 60.9% (vs 59.4%) | 57.4% | 0.643 | 39.3% | 75.6% |

*Note: NYU is exploratory ($N=6$, 2 Control, 4 ADHD). Peking combines Peking 1, 2, and 3.*


---

### 4. What Configurations Won Inside the Training Folds?
| Held-out Site | Winner Selected by Inner 3-Fold CV | Inner 3-Fold AUC |
|:---|:---|:---:|
| **KKI** | `Ensemble_Top3` | 0.6772 |
| **NYU** | `pearson__k50__covTrue__LogReg(C=0.1, cw=balanced)` | 0.6472 |
| **NeuroIMAGE** | `Ensemble_Top3` | 0.6695 |
| **Peking** | `pearson__k50__covFalse__LinearSVM(C=0.01, cw=balanced)` | 0.6111 |

---

### 5. Rigorous Statistical Significance & Hypothesis Testing
#### A. Within-Institution Label Permutation Test
- **Procedure**: Labels permuted **strictly within each institution** across 200 shuffles.
- **Full Pipeline Re-selection**: Each permutation re-ran the full nested inner 3-fold cross-validation across all 126 candidates to select the fold winner.
- **Final Candidate Observed LOIO ROC-AUC**: **0.6253**
- **Empirical Null Distribution**: Mean = 0.4881 ± 0.0547 (95th percentile = 0.5831, 99th percentile = 0.6130)
- **Empirical $p$-value**: **$p = 0.0050$**


#### B. Neuroimaging Connectivity vs Demographic Covariates (Age & Sex Alone)
- **Age & Sex Alone (LOIO ROC-AUC)**: **0.5857** ($p = 0.0498$ under within-site permutation).
- **Paired Bootstrap Test (2,000 Stratified Subject Resamples)**:
  - $\Delta\text{AUC} (\text{Final Candidate} - \text{Age/Sex Alone}) = +0.0400 [-0.1038, +0.1714]$
  - One-sided empirical $p$-value ($H_0: \Delta\text{AUC} \le 0$): **$p = 0.2685$**
  - Two-sided interpretation: The 95% bootstrap confidence interval spans zero [$-0.104, +0.171$], showing that connectivity does not achieve statistically significant superiority over demographic covariates alone.

---

### 6. Refitted Final Model Artifact
- **Global Winner (Inner 3-Fold Selection on All 118 Subjects)**: `Ensemble_Top3` (Inner CV AUC: 0.6168)
- **Persisted Artifact**: [`final_candidate_pipeline.joblib`](<repo>/NeuroADHD_ModelBenchmark/experiments/final_candidate/final_candidate_pipeline.joblib)
- **Verification**: Leakage unit tests in `test_candidate_leakage.py` passed with 0 failures.


---

### 7. Methodological Limitations & Model Selection Caveats
1. **Search Space Narrowing & Meta-Selection Confound**: The 126-configuration grid evaluated here was narrowed from an initial 576-configuration space based on findings from earlier exploratory studies on the exact same 118-subject LOIO cohort. Because the parameter ranges (selecting balanced weights, $k \in \{50, 100, 200\}$, and pruning unregularized models) were informed by data from these same subjects, this introduces an exploratory selection bias.
2. **Permutation Test Scope**: The 200 within-institution label permutations re-ran the full nested inner 3-fold cross-validation across all 126 configurations. However, the permutation procedure **does not repeat the outer narrowing from 576 to 126 configurations**. The empirical null distribution therefore conditions on the pre-narrowed search space.
3. **Non-Significant Separation from Demographic Baselines**: While the final connectivity candidate achieves a statistically significant LOIO ROC-AUC of 0.6253 against chance ($p = 0.0050$), the paired bootstrap comparison against age and sex alone ($\Delta\text{AUC} = +0.0400$, $p = 0.2685$) demonstrates that resting-state functional connectivity does not significantly outperform demographic covariates alone on this cohort.
4. **Sample Size & Multi-Site Imbalance**: Even after excluding zero-ADHD sites (OHSU, Pittsburgh), the clean cohort comprises only 118 subjects across 4 institutions. NYU contains only 6 subjects (2 Control, 4 ADHD), and site-level base rates continue to vary across institutions.
