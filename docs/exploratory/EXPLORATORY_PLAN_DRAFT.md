# Exploratory Analysis Protocol Draft: Multi-Site Resting-State fMRI Connectivity in ADHD-200

> [!WARNING]
> **STRICTLY EXPLORATORY PROTOCOL — NO CONFIRMATORY CLAIMS**  
> This protocol specifies an exploratory, hypothesis-generating investigation across the five multi-class ADHD-200 institutions in the Athena preprocessing release.  
> **The sealed NYU holdout is unsealed and permanently exhausted** (unblinded and evaluated during Study 2 under OSF registration [https://osf.io/grbt2](https://osf.io/grbt2)). Consequently, **NYU is no longer an external test site**; it is incorporated into the open development and cross-validation pool.  
> **No confirmatory claims of generalizable diagnostic discrimination may be drawn from any run under this protocol.** Any observed performance gain is strictly hypothesis-generating. Any confirmatory claim of clinical utility or external generalization strictly requires a new, previously unexamined, and externally pre-registered independent clinical cohort.

---

## 1. Objectives & Methodological Scope

The objective of this exploratory protocol is to investigate whether methodological refinements—specifically Riemannian tangent space connectivity, edge-level motion residualization, regularized linear models, and multi-run temporal SNR aggregation—can recover diagnostic discrimination above chance and above non-imaging baselines under strictly nested, leakage-free validation.

### Core Methodological Guardrails
1. **Pre-Declared Configuration Budget ($12 + 4$ Grid):** To prevent post-hoc hyperparameter fishing, multiple testing inflation, and garden-of-forking-paths bias, the exploration is strictly capped at **16 pre-declared configurations** partitioned into:
   - **12 Primary Configurations:** Evaluated across the full 5-site cohort in Leave-One-Institution-Out (LOIO) cross-validation ($3 \text{ feature spaces} \times 2 \text{ feature selections} \times 2 \text{ classifiers}$).
   - **4 Multi-Run Configurations:** Evaluated as a separate, isolated sub-analysis on NYU and OHSU only (the only two sites with multiple runs), using subject-grouped folds.
2. **Full Outcome Disclosure (Reporting Every Tried Configuration):** Every executed configuration—regardless of whether it achieves high performance, underperforms chance, or fails completely—must be logged and reported in full in the final summary table. Selective reporting or cherry-picking of optimal folds or runs is strictly prohibited.
3. **Mandatory Non-Imaging Benchmarks:** Every evaluated configuration must be benchmarked against:
   - Dummy Majority-Class Baseline.
   - Logistic Regression on Rigid-Body Motion Summary Statistics (`mean_fd_mm`, `max_disp_mm`).
   - Logistic Regression on Demographic Covariates (`age`, biological `sex`).
   - Primary Combined Baseline: Logistic Regression on Motion + Age + Sex (`[mean_fd_mm, max_disp_mm, age, sex]`).
4. **Primary Exploratory Contrast:**
   $$\Delta \text{AUC}_{\text{incremental}} = \text{AUC}\Big(\text{Motion} + \text{Age} + \text{Sex} + \text{Connectivity}\Big) - \text{AUC}\Big(\text{Motion} + \text{Age} + \text{Sex}\Big)$$
   evaluated on identical cross-validation folds with a paired cluster-bootstrap (2,000 resamples at the subject level).
5. **Pre-Declared Criterion for an "Interesting" Finding:**
   To be declared "interesting" (worthy of further exploration in a new cohort), the top-ranking configuration must satisfy:
   - $\Delta \text{AUC}$ vs. Motion Baseline $> 0$ with 95% paired bootstrap CI lower bound $> 0$.
   - $\Delta \text{AUC}$ vs. Age/Sex Baseline $> 0$ with 95% paired bootstrap CI lower bound $> 0$.
   - $\Delta \text{AUC}_{\text{incremental}}$ (Motion + Age + Sex + Connectivity vs. Motion + Age + Sex) $> 0$ with 95% paired bootstrap CI lower bound $> 0$.
   If the lower bound of any of these intervals encompasses zero, the connectivity features provide no statistically credible incremental utility beyond non-imaging confounders.
6. **Zero Individual Data Leakage:** Reports, tables, logs, and artifacts must never print individual subject IDs, raw timeseries, or per-subject prediction rows.

---

## 2. Cohort Stratification & Cross-Validation Architecture

### 2.1 Included Multi-Class Institutions ($N = 620$ Scanned Subjects)
The 5 multi-class institutions with resting-state CC200 functional time courses and motion transform matrices:
- **Peking University** (Peking 1, 2, 3): $N = 194$ (116 TDC, 78 ADHD)
- **Kennedy Krieger Institute** (KKI): $N = 83$ (61 TDC, 22 ADHD)
- **NeuroIMAGE**: $N = 48$ (23 TDC, 25 ADHD)
- **Oregon Health & Science University** (OHSU): $N = 79$ (42 TDC, 37 ADHD)
- **New York University Child Study Center** (NYU): $N = 216$ (98 TDC, 118 ADHD) — *treated strictly as an open development site*

### 2.2 Quality Censoring: Section 5.3 Motion Exclusion
Following Study 2 pre-registration Section 5.3, subjects with mean framewise displacement $\text{mean FD} > 0.50\text{ mm}$ or $> 20\%$ frames exceeding $0.50\text{ mm}$ are excluded in the clean primary cohort:
- **Clean Primary Retained Cohort ($N = 564$):** Peking 189, KKI 79, NeuroIMAGE 42, OHSU 58, NYU 196 (321 TDC, 243 ADHD).
- **Excluded Scans ($N = 56$):** Peking 5, KKI 4, NeuroIMAGE 6, OHSU 21, NYU 20 (disproportionately ADHD: $14.8\%$ ADHD vs. $5.0\%$ TDC, Fisher $p = 0.0011$).
Both the clean primary cohort ($N = 564$) and full scanned cohort ($N = 620$) are evaluated to inspect motion censoring sensitivity.

### 2.3 Excluded Institutions ($N = 353$ Subjects)
- **Pittsburgh** ($N = 98$ phenotype rows, $N = 89$ archive subjects): 100% control in CC200 release (0 ADHD cases in imaging archive; single-class site).
- **WashU** ($N = 61$ phenotype rows, $N = 61$ archive subjects): 100% control in CC200 release (single-class site). All WashU runs (including 44 `rest_2` and 36 `rest_3`) are excluded from modeling.
- **Brown** ($N = 26$ phenotype rows): Unlabeled site (DX pending).
- **NITRC Archive Omissions** ($N = 162$ subjects across Peking, KKI, NeuroIMAGE, OHSU, NYU): Missing time courses.
- **Missing Motion Data** ($N = 6$ subjects at NYU): Missing motion matrices in `intermediate_nyu.tar`.

### 2.4 Cross-Validation Scheme: 5-Fold Leave-One-Institution-Out (LOIO)
- Outer folds: In each fold, 4 institutions serve as training, and the 5th institution is the unseen test set.
- All preprocessing parameters, feature selection thresholds, reference Fréchet covariance means, and model parameters are fit strictly on training institutions and applied out-of-sample to the held-out institution.

---

## 3. Motion-Matched Evaluation Protocol

Because head motion is significantly elevated in ADHD subjects, diagnostic classifiers can exploit residual motion patterns rather than neurobiological connectivity. To isolate neural connectivity signal:

1. **Matching Procedure:**
   - 1:1 nearest-neighbor matching without replacement between ADHD and TDC subjects within each institution, based on `mean_fd_mm` with a caliper distance $\le 0.05\text{ mm}$.
2. **Primary Motion Matching (Performed AFTER Section 5.3 FD Exclusion):**
   - Matching within the quality-controlled clean primary cohort ($N = 564$) ensures all evaluated subjects have clean scans and strictly balanced motion distributions.
   - **Resulting Cohort: 215 Matched Pairs ($N = 430$ Subjects: 215 TDC, 215 ADHD):**
     - **Peking:** 73 matched pairs ($N = 146$; 73 TDC, 73 ADHD)
     - **KKI:** 19 matched pairs ($N = 38$; 19 TDC, 19 ADHD)
     - **NeuroIMAGE:** 16 matched pairs ($N = 32$; 16 TDC, 16 ADHD)
     - **OHSU:** 21 matched pairs ($N = 42$; 21 TDC, 21 ADHD)
     - **NYU:** 86 matched pairs ($N = 172$; 86 TDC, 86 ADHD)
3. **Sensitivity Motion Matching (Performed BEFORE Section 5.3 FD Exclusion):**
   - Matching across all $N = 620$ scanned subjects:
   - **Resulting Cohort: 226 Matched Pairs ($N = 452$ Subjects: 226 TDC, 226 ADHD):**
     - Peking: 73 pairs ($N = 146$), KKI: 20 pairs ($N = 40$), NeuroIMAGE: 16 pairs ($N = 32$), OHSU: 28 pairs ($N = 56$), NYU: 89 pairs ($N = 178$).
4. **Reporting Strategy:**
   - The primary evaluation reports both the **unmatched analysis** ($N = 564$ clean, $N = 620$ scanned) and the **motion-matched analysis** ($N = 430$) side-by-side to directly measure the performance drop under motion control.

---

## 4. Candidate Model Configurations ($12 + 4$ Factorial Grid)

### 4.1 The 12 Primary 5-Site LOIO Configurations
Evaluated across all 5 institutions ($3 \text{ Feature Kinds} \times 2 \text{ Feature Selections} \times 2 \text{ Classifiers}$):

#### Dimension A: Feature Representation & Motion Handling (3 Kinds)
1. **`A1` (Pearson + Fisher $z$):**
   - Pairwise sample Pearson correlation matrix $R$ across 190 CC200 ROIs, clipped to $[-1 + 10^{-7}, 1 - 10^{-7}]$, Fisher-$z$ transformed ($z = \operatorname{arctanh}(r)$).
   - Off-diagonal upper triangle: $p = 190 \times 189 / 2 = 17,955$ features.
2. **`A2` (Riemannian Tangent Space Connectivity):**
   - Ledoit-Wolf shrinkage covariance matrices $S_i$ estimated per subject.
   - Fréchet geometric mean covariance $\bar{\Sigma}$ estimated **strictly on training fold subjects** on the manifold of symmetric positive definite (SPD) matrices.
   - Manifold tangent projection: $\Omega_i = \operatorname{Logm}(\bar{\Sigma}^{-1/2} S_i \bar{\Sigma}^{-1/2})$.
   - State vector length:
     - **With diagonal (standard Riemannian isometry):** $p = 190 \times 191 / 2 = 18,145$ features (diagonal elements retained, off-diagonals scaled by $\sqrt{2}$ to preserve the Riemannian manifold metric in Euclidean space, matching `nilearn.connectome.ConnectivityMeasure(kind='tangent', vectorize=True)`).
     - **Without diagonal:** $p = 17,955$ features (strictly off-diagonal tangent coordinates).
     - Primary specification uses the standard isometry with diagonal ($p = 18,145$).
3. **`A3` (Pearson + Motion-Covariate Residualization):**
   - Edge-level ordinary least squares (OLS) regression fit **strictly on training fold subjects**:
     $$z_{ij} = \beta_0 + \beta_1 \cdot \text{mean\_fd\_mm} + \beta_2 \cdot \text{max\_disp\_mm} + \epsilon_{ij}$$
   - Regression weights applied out-of-sample to compute residual connectivity $z_{ij}^{\text{resid}} = z_{ij} - \hat{z}_{ij}$ on both training and test folds. Vector length: $p = 17,955$.

#### Dimension B: Feature Selection (2 Levels)
1. **`B1` (ANOVA $k=100$):** Univariate feature selection retaining the top $k = 100$ features with highest $F$-statistic, fit strictly on training fold labels.
2. **`B2` (All Edges):** Zero feature selection; all edges retained ($17,955$ or $18,145$).

#### Dimension C: Classifier Architecture (2 Levels)
1. **`C1` (Balanced Random Forest):**
   - `sklearn.ensemble.RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42)` (identical to locked Study 1/2 classifier).
2. **`C2` ($L_2$ Regularized Logistic Regression):**
   - Fixed penalty parameter stated in advance: $C = 1.0$ (no post-hoc hyperparameter optimization).
   - `sklearn.linear_model.LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)`.

---

### 4.2 The 4 Multi-Run Configurations (NYU + OHSU Sub-Analysis)
Because only NYU ($180$ `rest_2` runs) and OHSU ($78$ `rest_2`, $78$ `rest_3` runs) acquired multiple resting-state sessions, multi-run aggregation is evaluated as an **isolated, exploratory sub-analysis**:
- **Design:** Averaging Fisher-$z$ Pearson correlation matrices across all available runs of each subject, boosting temporal scan duration ($T_{\text{eff}} \approx 2\times\text{--}3\times$) and SNR.
- **Subject-Grouped Cross-Validation:** Folds are strictly grouped by unique subject ID; all runs of a subject remain in the same split. Effective sample size $N$ remains the number of unique subjects (one feature vector and one label row per subject).
- **Non-Comparability Statement:** This sub-analysis is **not directly comparable** across the 5 sites because KKI, NeuroIMAGE, and Peking have only single runs. WashU runs (44 `rest_2`, 36 `rest_3`) are excluded because WashU has 0 ADHD cases (single-class).

---

### 4.3 Complete Enumeration Table ($12 + 4 = 16$ Configurations)

| Config ID | Target Scope | Feature Representation | Feature Selection | Classifier Architecture | Input Data & Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`CFG_01`** | 5-Site LOIO | Pearson + Fisher $z$ ($p=17,955$) | ANOVA ($k=100$) | Balanced Random Forest | 190 ROIs, `rest_1` (Study 2 locked model) |
| **`CFG_02`** | 5-Site LOIO | Pearson + Fisher $z$ ($p=17,955$) | ANOVA ($k=100$) | $L_2$ Logistic Regression ($C=1.0$) | 190 ROIs, `rest_1` |
| **`CFG_03`** | 5-Site LOIO | Pearson + Fisher $z$ ($p=17,955$) | All Edges ($p=17,955$) | Balanced Random Forest | 190 ROIs, `rest_1` |
| **`CFG_04`** | 5-Site LOIO | Pearson + Fisher $z$ ($p=17,955$) | All Edges ($p=17,955$) | $L_2$ Logistic Regression ($C=1.0$) | 190 ROIs, `rest_1` |
| **`CFG_05`** | 5-Site LOIO | Tangent Space ($p=18,145$) | ANOVA ($k=100$) | Balanced Random Forest | Ledoit-Wolf, Fréchet mean fit on train |
| **`CFG_06`** | 5-Site LOIO | Tangent Space ($p=18,145$) | ANOVA ($k=100$) | $L_2$ Logistic Regression ($C=1.0$) | Ledoit-Wolf, Fréchet mean fit on train |
| **`CFG_07`** | 5-Site LOIO | Tangent Space ($p=18,145$) | All Edges ($p=18,145$) | Balanced Random Forest | Ledoit-Wolf, Fréchet mean fit on train |
| **`CFG_08`** | 5-Site LOIO | Tangent Space ($p=18,145$) | All Edges ($p=18,145$) | $L_2$ Logistic Regression ($C=1.0$) | Ledoit-Wolf, Fréchet mean fit on train |
| **`CFG_09`** | 5-Site LOIO | Motion Residualized ($p=17,955$) | ANOVA ($k=100$) | Balanced Random Forest | OLS motion weights fit on train |
| **`CFG_10`** | 5-Site LOIO | Motion Residualized ($p=17,955$) | ANOVA ($k=100$) | $L_2$ Logistic Regression ($C=1.0$) | OLS motion weights fit on train |
| **`CFG_11`** | 5-Site LOIO | Motion Residualized ($p=17,955$) | All Edges ($p=17,955$) | Balanced Random Forest | OLS motion weights fit on train |
| **`CFG_12`** | 5-Site LOIO | Motion Residualized ($p=17,955$) | All Edges ($p=17,955$) | $L_2$ Logistic Regression ($C=1.0$) | OLS motion weights fit on train |
| **`CFG_13`** | NYU+OHSU Sub | Multi-Run Averaged ($p=17,955$) | ANOVA ($k=100$) | Balanced Random Forest | Subject-grouped folds, WashU excluded |
| **`CFG_14`** | NYU+OHSU Sub | Multi-Run Averaged ($p=17,955$) | ANOVA ($k=100$) | $L_2$ Logistic Regression ($C=1.0$) | Subject-grouped folds, WashU excluded |
| **`CFG_15`** | NYU+OHSU Sub | Multi-Run Averaged ($p=17,955$) | All Edges ($p=17,955$) | Balanced Random Forest | Subject-grouped folds, WashU excluded |
| **`CFG_16`** | NYU+OHSU Sub | Multi-Run Averaged ($p=17,955$) | All Edges ($p=17,955$) | $L_2$ Logistic Regression ($C=1.0$) | Subject-grouped folds, WashU excluded |

---

## 5. Statistical Inference & Full-Selection Permutation Testing

### 5.1 Max-Statistic Within-Institution Permutation Test
To control the family-wise error rate across all candidate configurations ("winner's curse"):
1. Diagnostic labels ($y$) are permuted **strictly within each institution**, preserving site sample sizes, class balance, and site-scanner distributions.
2. In each permutation iteration $b = 1, \dots, B$:
   - All 12 primary LOIO configurations are re-fit and evaluated on the permuted labels.
   - The maximum LOIO ROC-AUC across all configurations is recorded: $\text{AUC}_{\max}^{(b)} = \max_{c} \text{AUC}_c^{(b)}$.
3. The empirical family-wise $p$-value for the observed winning configuration $\text{AUC}_{\max}^{\text{obs}}$ is:
   $$p_{\text{perm}} = \frac{1 + \sum_{b=1}^{B} \mathbb{I}\left(\text{AUC}_{\max}^{(b)} \ge \text{AUC}_{\max}^{\text{obs}}\right)}{1 + B}$$

### 5.2 Permutation Budget & Runtime Estimates
Benchmark measurements on a full synthetic 5-site cohort ($N = 564, p = 17,955$):
- 1 complete 5-fold LOIO evaluation across all 12 configurations: **$2.70\text{ seconds}$**.
- **200 Permutations:** Estimated runtime $\approx \mathbf{9.0\text{ minutes}}$ (recommended operational default).
- **1,000 Permutations:** Estimated runtime $\approx \mathbf{45.0\text{ minutes}}$.
The protocol specifies $B = 200$ iterations as primary, with $B = 1,000$ optional if compute budget allows.

---

## 6. Output Reporting Standards

The final summary report must tabulate:
1. **Pooled LOIO Metrics:** Pooled ROC-AUC (95% CI), Balanced Accuracy (95% CI), $\Delta \text{AUC}$ vs. Motion Baseline, $\Delta \text{AUC}$ vs. Age/Sex Baseline, Primary Contrast $\Delta \text{AUC}_{\text{incremental}}$ (Motion+Age+Sex+Connectivity vs. Motion+Age+Sex), and Permutation $p$-value.
2. **Per-Institution Metrics:** Individual held-out ROC-AUC and 95% CIs for Peking, KKI, NeuroIMAGE, OHSU, and NYU.
3. **Motion-Matched vs. Unmatched:** Direct side-by-side comparison across all metrics.
4. **Mandatory Concluding Clause:**
   > *"These exploratory findings represent post-hoc analyses on previously analyzed data and do not establish clinical validity. Confirmatory testing requires pre-registration and replication on a novel, untouched external clinical cohort."*
