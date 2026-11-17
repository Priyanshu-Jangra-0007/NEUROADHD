# Pre-Registration: Study 2 - Multi-Site Generalization on Athena Derivatives

**Registration Status:** Frozen & Committed Pre-Registration Protocol  
**Date:** November 17, 2026  
**Execution Condition:** This document is committed to version control *prior* to downloading, inspecting, or processing any Athena derivative files.  

---

## 1. Study Objective & Scientific Question

The primary objective of Study 2 is to evaluate whether the locked functional connectivity classification pipeline developed in Study 1 generalizes across a multi-site resting-state fMRI cohort without data leakage, and whether resting-state functional connectivity provides diagnostic classification beyond head motion and demographic covariates (age and sex).

---

## 2. Locked Model Pipeline Specification

The classification pipeline is frozen a priori and refitted from scratch within each training fold (no pre-fitted scalers, imputers, or feature selectors):

1. **Input Features:** Upper-triangle pairwise Fisher $z$-transformed Pearson correlation coefficients computed from regional BOLD time series:
   $$z = \frac{1}{2} \ln\left(\frac{1 + r}{1 - r}\right)$$
   For $K$ valid ROIs, this yields $P = \frac{K(K - 1)}{2}$ functional connectivity edges (e.g., 19,900 edges for $K = 200$).
2. **Step 1 - Missing Value Imputation:** `SimpleImputer(strategy='median')`
3. **Step 2 - Zero-Variance Filter:** `VarianceThreshold(threshold=0.0)`
4. **Step 3 - Feature Standardization:** `StandardScaler()`
5. **Step 4 - Feature Selection:** `SelectKBest(score_func=f_classif, k=100)`
6. **Step 5 - Estimator:**
   ```python
   RandomForestClassifier(
       n_estimators=50,
       max_depth=3,
       class_weight='balanced',
       random_state=42
   )
   ```

All hyperparameters, random seeds, and filtering criteria are strictly locked. No hyperparameter searches or candidate model exploration will be performed.

---

## 3. Cohort Definition, Pre-Registered Partitions, and Sealing Rule

### 3.1. Written Rule for Sealing an Institution
To ensure an untainted external validation test, one institution must be completely sealed until all development models, motion thresholds, and baseline comparisons are finalized. The sealed institution is selected according to the following pre-registered rule:
> *The sealed institution must have balanced representation of both diagnostic classes (>5% ADHD), standard pediatric acquisition parameters, and minimal or zero exposure during historical pipeline exploration.*

* **Selected Sealed Institution:** **NYU** ([UNVERIFIED] phenotypic count: $N = 216$: 98 TDC, 118 ADHD). Only 6 NYU subjects were included in the historical 153-subject C-PAC cohort.
* **Evaluation Condition:** NYU will be evaluated **strictly once**, after the LOIO development protocol and all statistical tests on the remaining sites are completely executed and frozen.

### 3.2. Leave-One-Institution-Out (LOIO) Development Cohort
Cross-validation will be conducted using Leave-One-Institution-Out (LOIO) across the remaining 4 multi-class institutions:
* **KKI:** [UNVERIFIED] $N = 83$ (61 TDC, 22 ADHD)
* **NeuroIMAGE:** [UNVERIFIED] $N = 48$ (23 TDC, 25 ADHD)
* **OHSU:** [UNVERIFIED] $N = 79$ (42 TDC, 37 ADHD). *Note: 14 OHSU controls were present in historical C-PAC exploration.*
* **Peking:** [UNVERIFIED] $N = 194$ (116 TDC, 78 ADHD across Peking 1, 2, 3)

**Cohort Counts Notice:**
* Development Cohort: **[UNVERIFIED] $N = 404$**
* Sealed Validation Cohort (NYU): **[UNVERIFIED] $N = 216$**
* Total Analyzed Cohort: **[UNVERIFIED] $N = 620$** (not 640)
* *All subject counts are cited from the ADHD-200 competition training release phenotype tables and remain strictly UNVERIFIED until physical archive files on disk are inspected and verified.*

### 3.3. Excluded Sites
Institutions lacking multi-class diagnostic representation in the training release are excluded a priori:
* **Pittsburgh:** [UNVERIFIED] $N = 89$ (100% TDC, 0% ADHD; excluded)
* **WashU:** [UNVERIFIED] $N = 59$ (100% TDC, 0% ADHD; excluded)
* **Brown:** [UNVERIFIED] $N = 8$ (Diagnostic labels unreleased in competition training set; excluded)

---

## 4. Quality Control & Motion Handling Rules

### 4.1. A Priori ROI Inclusion Rule
Because parcellation masks may not cover 100% of brain volume across all subjects, some ROIs may contain constant zero or missing signals:
* **Rule:** An ROI must possess non-zero variance across **100% of subjects** in the analyzed cohort. Any ROI that is constant or empty in even one subject is discarded across all subjects prior to computing the correlation matrix.

### 4.2. Power's Framewise Displacement (FD) Formula & Exclusion Threshold
Head motion will be quantified from the 6 rigid-body motion parameters ($r_x, r_y, r_z$ rotations in degrees; $t_x, t_y, t_z$ translations in millimeters) using Power et al. (2012) Framewise Displacement:
1. Rotations are converted to arc-length displacements on the surface of a sphere of radius $r = 50\,\text{mm}$:
   $$d_{\theta, i} = \theta_i \times \frac{\pi}{180} \times 50\,\text{mm}$$
2. Frame-to-frame displacement for frame $i$ is calculated as:
   $$\text{FD}_i = |\Delta t_{x,i}| + |\Delta t_{y,i}| + |\Delta t_{z,i}| + |\Delta d_{r_x,i}| + |\Delta d_{r_y,i}| + |\Delta d_{r_z,i}|$$
   where $\Delta x_i = x_i - x_{i-1}$ and $\text{FD}_0 = 0$.

* **Exclusion Threshold:** A scan is excluded if:
  $$\text{Mean FD} > 0.50\,\text{mm} \quad \text{OR} \quad \text{Percentage of frames with } \text{FD} > 0.50\,\text{mm} > 20\%$$
* **Attrition Reporting Rule:** The number of excluded participants must be reported broken down by site and diagnosis (TDC vs. ADHD) to verify that motion exclusion does not introduce differential attrition bias.

---

## 5. Baselines & Benchmark Comparators

To prevent deceptive conclusions arising from confounding variables, the locked pipeline will be compared against two non-imaging baselines evaluated under the exact same LOIO folds:
1. **Motion-Only Baseline:** A logistic regression classifier trained exclusively on head motion metrics ($\text{Mean FD}$ and $\text{Max Displacement}$) with balanced class weighting.
2. **Age & Sex Baseline:** A logistic regression classifier trained on standardized age and biological sex:
   ```python
   StandardScaler() -> LogisticRegression(C=1.0, class_weight='balanced')
   ```

---

## 6. Statistical Analysis & Endpoints

### 6.1. Primary Endpoint
* **Metric:** Leave-One-Institution-Out (LOIO) ROC-AUC across the development cohort ($N = 404$).
* **Significance Testing:** Non-parametric permutation test ($B = 1,000$ iterations). To preserve site-specific marginal prevalence and demographic structures, labels will be permuted **strictly within each institution**:
  $$p_{\text{perm}} = \frac{1 + \sum_{b=1}^B \mathbb{I}(\text{AUC}_b \ge \text{AUC}_{\text{observed}})}{1 + B}$$
* **Confidence Interval:** Stratified bootstrap 95% confidence interval ($B = 2,000$ resamples).

### 6.2. Secondary Endpoint & Statistical Power Caveat
* **Metric:** Incremental diagnostic value over non-imaging demographic covariates:
  $$\Delta\text{AUC} = \text{AUC}_{\text{fMRI}} - \text{AUC}_{\text{Age/Sex}}$$
  evaluated via paired stratified bootstrap.
* **Power Caveat:**
  > Assuming a true moderate increment of $\Delta\text{AUC} = +0.03\text{--}0.04$, statistical power to reject $H_0: \Delta\text{AUC} \le 0$ at two-sided $\alpha = 0.05$ is only **about 18% at $N = 404$** and **about 26% at $N = 620$**, due to substantial covariance between fMRI connectivity predictions and demographic characteristics. A non-significant $\Delta\text{AUC}$ cannot be interpreted as proof that functional connectivity carries no signal, but rather reflects the sample size constraints of the dataset.

---

## 7. Success Criteria & Scientific Interpretation

* **Development Success:** The locked pipeline is deemed promising in development if:
  1. LOIO ROC-AUC significantly exceeds chance ($p_{\text{perm}} < 0.05$).
  2. LOIO ROC-AUC exceeds the motion-only baseline.
* **Sealed External Confirmation:** The locked pipeline achieves confirmed generalization if:
  1. NYU sealed ROC-AUC exceeds $0.55$, with a 95% bootstrap confidence interval lower bound $> 0.50$.
* **Conclusion if Sealed NYU AUC is Near 0.50:**
  > *If the locked model achieves near-chance performance ($\text{AUC} \approx 0.50$) on the sealed NYU cohort, we will explicitly conclude that resting-state fMRI functional connectivity under this pipeline does not provide generalizable out-of-site classification of ADHD. Any apparent cross-validation signal observed in smaller cohorts will be reported as reflecting site-specific scanner artifacts, demographic confounding, or sampling variability, rather than reproducible neurobiological markers.*

---

## 8. Atlas Divergence Acknowledgment

The Athena pipeline derivatives were generated via AFNI using an AFNI MNI template space, whereas Study 1 utilized C-PAC derivatives coregistered to the Craddock 2012 CC200 NIfTI template in FSL space. Due to potential discrepancies in coordinate grids, bounding boxes, and ROI indexing, **the Athena CC200 atlas cannot be assumed identical to the Study 1 CC200 atlas**. Study 2 is an independent multi-site evaluation, and quantitative results are not directly comparable to Study 1.
