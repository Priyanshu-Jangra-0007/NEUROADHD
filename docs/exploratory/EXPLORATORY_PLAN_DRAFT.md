# Exploratory Analysis Protocol Draft: Multi-Site Resting-State fMRI Connectivity in ADHD-200

> [!WARNING]
> **STRICTLY EXPLORATORY PROTOCOL — NO CONFIRMATORY CLAIMS**  
> This protocol specifies an exploratory, hypothesis-generating investigation across the five multi-class ADHD-200 institutions in the Athena preprocessing release.  
> **The sealed NYU holdout is unsealed and permanently exhausted** (unblinded and evaluated during Study 2 under OSF registration [https://osf.io/grbt2](https://osf.io/grbt2)). Consequently, **NYU is no longer an external test site**; it is incorporated into the open development and cross-validation pool.  
> **No confirmatory claims of generalizable diagnostic discrimination may be drawn from any run under this protocol.** Any observed performance gain is strictly hypothesis-generating. Any confirmatory claim of clinical utility or external generalization strictly requires a new, previously unexamined, and externally pre-registered independent clinical cohort.  
> **Dashboard Integration Policy:** Results from this exploratory protocol do NOT go on the production research dashboard unless placed in a dedicated panel explicitly labeled *"Exploratory / Hypothesis-Generating Follow-Up"*.

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
4. **Primary Incremental Exploratory Contrast:**
   $$\Delta \text{AUC}_{\text{incremental}} = \text{AUC}\Big(\text{Motion} + \text{Age} + \text{Sex} + \text{Connectivity}\Big) - \text{AUC}\Big(\text{Motion} + \text{Age} + \text{Sex}\Big)$$
   evaluated on identical cross-validation folds with a paired cluster-bootstrap (2,000 resamples at the subject level).
5. **Pre-Declared Criteria for an "Interesting" Finding:**
   To be declared "interesting" (worthy of further exploration in a new cohort), the top-ranking configuration must satisfy:
   - **Max-Statistic Incremental Contrast Permutation Test:** $p_{\text{incremental}} < 0.05$ under the connectivity-row permutation null (repeating full selection over the 12 configurations).
   - **Paired Bootstrap Confidence Interval:** 95% paired bootstrap CI lower bound for $\Delta \text{AUC}_{\text{incremental}} > 0$.
     *(Note: This confidence interval is conditional on configuration selection and is thus subject to selection optimism / winner's curse; the permutation test provides the formal family-wise control).*
   - **Benchmark Superiority:** Empirical point estimates $\Delta \text{AUC} > 0$ relative to both the motion-only baseline and the age/sex baseline.
   If any of these conditions fails, the connectivity features provide no statistically credible incremental utility beyond non-imaging confounders.
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

## 3. Deterministic Motion-Matched Evaluation Protocol

Because head motion is significantly elevated in ADHD subjects, diagnostic classifiers can exploit residual motion patterns rather than neurobiological connectivity. To isolate neural connectivity signal, an explicit 1:1 motion-matched subset is constructed.

### 3.1 Fully Deterministic Matching Algorithm
To eliminate order dependency and guarantee bit-for-bit reproducibility:
1. Cohort rows are pre-filtered to the target evaluation set (clean primary $N = 564$ or full scanned $N = 620$).
2. Within each of the 5 institutions:
   - ADHD subjects are sorted deterministically by `(mean_fd_mm ascending, normalized_subject_id ascending)`.
   - Candidate TDC subjects are sorted deterministically by `(mean_fd_mm ascending, normalized_subject_id ascending)`.
   - The algorithm iterates through sorted ADHD subjects one by one.
   - For each ADHD subject, candidate TDC subjects within a maximum caliper distance of $|FD_{\text{TDC}} - FD_{\text{ADHD}}| \le 0.05\text{ mm}$ are evaluated.
   - **Tie-Breaking Rule:** From all eligible TDC candidates, the match is chosen by:
     a) Minimum absolute distance $|FD_{\text{TDC}} - FD_{\text{ADHD}}|$.
     b) If distances are tied, candidate with smaller $FD_{\text{TDC}}$.
     c) If still tied, candidate with lexicographically smaller normalized `subject_id`.
   - The selected TDC subject is assigned and removed from the available pool.
   - Entire procedure contains zero stochastic components.

### 3.2 Explanation of the 227 vs. 226 Pre-Exclusion Count
- In greedy nearest-neighbor matching without optimal bipartite matching (Hungarian assignment), match cardinality is sensitive to the processing order of the source subjects.
- When phenotypic rows were loaded in unsorted iteration order from concatenated per-site CSVs, a specific ADHD subject in Peking and NeuroIMAGE matched a TDC subject that allowed another ADHD subject to also find a match within caliper (yielding 227 pairs).
- When records were loaded in file order from `adhd200_preprocessed_phenotypics.tsv`, an earlier ADHD subject claimed that TDC subject, leaving a subsequent ADHD subject without any candidate within the 0.05 mm caliper (yielding 226 pairs).
- Specifying deterministic sorting by `(mean_fd_mm, normalized_subject_id)` locks the algorithm to an unambiguous, unique count.

### 3.3 Resulting Matched Cohorts
- **Primary Matching (AFTER Section 5.3 FD Exclusion, $N = 564$):**
  - **215 Matched Pairs ($N = 430$ Subjects: 215 TDC, 215 ADHD):**
    - Peking: 73 pairs ($N = 146$), KKI: 19 pairs ($N = 38$), NeuroIMAGE: 16 pairs ($N = 32$), OHSU: 21 pairs ($N = 42$), NYU: 86 pairs ($N = 172$).
- **Sensitivity Matching (BEFORE Section 5.3 FD Exclusion, $N = 620$):**
  - **226 Matched Pairs ($N = 452$ Subjects: 226 TDC, 226 ADHD):**
    - Peking: 73 pairs ($N = 146$), KKI: 20 pairs ($N = 40$), NeuroIMAGE: 16 pairs ($N = 32$), OHSU: 28 pairs ($N = 56$), NYU: 89 pairs ($N = 178$).

---

## 4. Candidate Model Configurations ($12 + 4$ Factorial Grid)

### 4.1 The 12 Primary 5-Site LOIO Configurations
Evaluated across all 5 institutions ($3 \text{ Feature Kinds} \times 2 \text{ Feature Selections} \times 2 \text{ Classifiers}$):

#### Dimension A: Feature Representation & Motion Handling (3 Kinds)
1. **`A1` (Pearson + Fisher $z$):**
   - Pairwise sample Pearson correlation matrix $R$ across 190 CC200 ROIs, clipped to $[-1 + 10^{-7}, 1 - 10^{-7}]$, Fisher-$z$ transformed ($z = \operatorname{arctanh}(r)$).
   - Off-diagonal upper triangle: $p = 190 \times 189 / 2 = 17,955$ features.
2. **`A2` (Riemannian Tangent Space Connectivity):**
   - **Shrinkage Rule:** Ledoit-Wolf shrinkage covariance estimator ($\ell_2$-regularized convex combination between sample covariance and identity/spherical shrinkage target).
   - **Fréchet Mean:** Fréchet geometric mean covariance $\bar{\Sigma}$ estimated **strictly on training fold subjects** on the manifold of symmetric positive definite (SPD) matrices.
   - **Manifold Projection:** $\Omega_i = \operatorname{Logm}(\bar{\Sigma}^{-1/2} S_i \bar{\Sigma}^{-1/2})$.
   - **Pre-Specified Vector Length:** $p = 190 \times 191 / 2 = 18,145$ features (retaining diagonal elements and scaling off-diagonals by $\sqrt{2}$ to preserve the Riemannian manifold metric in Euclidean space, matching `nilearn.connectome.ConnectivityMeasure(kind='tangent', vectorize=True)`).
3. **`A3` (Pearson + Motion-Covariate Residualization):**
   - Edge-level ordinary least squares (OLS) regression fit **strictly on training fold subjects**:
     $$z_{ij} = \beta_0 + \beta_1 \cdot \text{mean\_fd\_mm} + \beta_2 \cdot \text{max\_disp\_mm} + \epsilon_{ij}$$
   - Regression weights applied out-of-sample to compute residual connectivity $z_{ij}^{\text{resid}} = z_{ij} - \hat{z}_{ij}$ on both training and test folds ($p = 17,955$).

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

### 4.2 Limitation: Scan Length Heterogeneity & Rank Deficiency at OHSU ($T = 78 < K = 190$)
A critical neuroimaging limitation must be formally recognized:
- At OHSU, resting-state fMRI runs have only $T = 78$ retained volumes after dropping dummy frames, whereas the CC200 brain atlas contains $K = 190$ populated ROIs.
- Because $T = 78 < K = 190$, the empirical sample covariance matrix $S_i$ is **mathematically singular** ($\text{rank}(S_i) \le 77$).
- Ledoit-Wolf shrinkage regularization is strictly mandatory to pull the singular covariance toward a well-conditioned spherical target, ensuring full rank and invertibility prior to the matrix logarithm.
- However, the geometric covariance estimation on $T = 78$ is substantially noisier and less constrained than at sites with $T > 190$ (e.g. Peking $T = 233$, NeuroIMAGE $T = 257$).

---

### 4.3 The 4 Multi-Run Configurations (NYU + OHSU Sub-Analysis)
Because only NYU ($180$ `rest_2` runs) and OHSU ($78$ `rest_2`, $78$ `rest_3` runs) acquired multiple resting-state sessions, multi-run aggregation is evaluated as an **isolated, exploratory sub-analysis**:
- **Design:** Averaging Fisher-$z$ Pearson correlation matrices across all available runs of each subject, boosting temporal scan duration ($T_{\text{eff}} \approx 2\times\text{--}3\times$) and SNR.
- **Subject-Grouped Cross-Validation:** Folds are strictly grouped by unique subject ID; all runs of a subject remain in the same split. Effective sample size $N$ remains the number of unique subjects (one feature vector and one label row per subject).
- **Non-Comparability Statement:** This sub-analysis is **not directly comparable** across the 5 sites because KKI, NeuroIMAGE, and Peking have only single runs. WashU runs (44 `rest_2`, 36 `rest_3`) are excluded because WashU has 0 ADHD cases (single-class).

---

### 4.4 Complete Enumeration Table ($12 + 4 = 16$ Configurations)

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

## 5. Statistical Inference & Two Permutation Tests

Two distinct non-parametric permutation tests are conducted to evaluate performance:

### 5.1 Test 1: Label Permutation for Discrimination (H0: AUC = 0.5)
- **Null Hypothesis:** Diagnosis labels ($y$) are uncoupled from imaging and non-imaging features.
- Labels are permuted strictly within each institution.
- In each permutation $b = 1, \dots, B$, all 12 candidate configurations are re-fit, and the maximum LOIO ROC-AUC is recorded: $\text{AUC}_{\max}^{(b)} = \max_c \text{AUC}_c^{(b)}$.
- Family-wise empirical $p$-value:
  $$p_{\text{label}} = \frac{1 + \sum_{b=1}^{B} \mathbb{I}\left(\text{AUC}_{\max}^{(b)} \ge \text{AUC}_{\max}^{\text{obs}}\right)}{1 + B}$$

### 5.2 Test 2: Connectivity-Row Permutation for Incremental Contrast (H0: $\Delta\text{AUC}_{\text{incremental}} \le 0$)
- **Null Hypothesis:** Connectivity features carry zero diagnostic information beyond motion, age, and sex covariates.
- **Specification:** Diagnostic labels ($y$) and non-imaging covariates (`motion + age + sex`) stay **strictly paired and intact**.
- Only the **connectivity feature rows** ($X$) are permuted among subjects **strictly within each institution**.
- In each permutation $b = 1, \dots, B$:
  - The combined model `[motion + age + sex + connectivity_perm]` is evaluated on identical LOIO folds.
  - The incremental contrast $\Delta \text{AUC}_c^{(b)} = \text{AUC}_{\text{combined}, c}^{(b)} - \text{AUC}_{\text{baseline}}$ is computed relative to the fixed non-imaging baseline.
  - The maximum delta-AUC across all 12 configurations is recorded: $\Delta \text{AUC}_{\max}^{(b)} = \max_c \Delta \text{AUC}_c^{(b)}$.
- Family-wise empirical $p$-value:
  $$p_{\text{incremental}} = \frac{1 + \sum_{b=1}^{B} \mathbb{I}\left(\Delta \text{AUC}_{\max}^{(b)} \ge \Delta \text{AUC}_{\max}^{\text{obs}}\right)}{1 + B}$$

---

## 6. Real-Shape Benchmark & Execution Budget

Benchmarking on synthetic data at the exact clean primary dimensions ($N = 564, K = 190$, exact scan lengths $T \in [78, 257]$):
- **Tangent Feature Construction:** $3.75\text{ s}$ per fold ($18.75\text{ s}$ across all 5 folds).
- **Motion Residualization (OLS):** $0.055\text{ s}$ per fold.
- **Single Fold Model Fitting & Scoring (All 12 Configurations):** $0.861\text{ s}$ per fold.
- **Single Complete 5-Fold LOIO Evaluation:** $4.30\text{ s}$.
- **Permutation Cost with Cached Features:** **$3.58\text{ s}$ per iteration**.
- **200 Permutations Total:** $\approx \mathbf{11.9\text{ minutes}}$ ($\approx 716\text{ s}$).
- **1,000 Permutations Total:** $\approx \mathbf{59.7\text{ minutes}}$ ($\approx 3,581\text{ s}$).
*(All timings labeled as estimates).* Both 200 and 1,000 permutations fall comfortably below the 3-hour budget threshold.
