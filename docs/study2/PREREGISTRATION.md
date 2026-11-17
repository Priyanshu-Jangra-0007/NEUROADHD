# Pre-Registration: Study 2 - Multi-Site Generalization on Athena Derivatives

**Registration Status:** Frozen Pre-Registration Protocol (incorporating Amendment 2)  
**Date:** November 17, 2026  
**Execution Condition:** This protocol is committed to version control *prior* to downloading, inspecting, or analyzing any Athena derivative files.  

---

## 1. Study Objective & Scientific Question

The primary objective of Study 2 is to evaluate whether the locked functional connectivity classification pipeline developed in Study 1 generalizes across a multi-site resting-state fMRI cohort without data leakage, and whether resting-state functional connectivity provides diagnostic classification beyond head motion and demographic covariates (age and sex).

---

## 2. Locked Model Pipeline Specification

The classification pipeline is frozen a priori and refitted from scratch within each training fold (no pre-fitted scalers, imputers, or feature selectors):

1. **Feature Extraction:** Pairwise Pearson correlation followed by Fisher $z$-transformation, exactly as implemented in Study 1:
   $$r_{jk} = \frac{\sum_{t=1}^T (x_{jt} - \bar{x}_j)(x_{kt} - \bar{x}_k)}{\sqrt{\sum_{t=1}^T (x_{jt} - \bar{x}_j)^2 \sum_{t=1}^T (x_{kt} - \bar{x}_k)^2}}, \quad z_{jk} = \frac{1}{2} \ln\left(\frac{1 + r_{jk}}{1 - r_{jk}}\right)$$
   For $K$ valid ROIs, this yields $P = \frac{K(K - 1)}{2}$ functional connectivity edges from the upper triangle.
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

## 3. Data File Selection & Preprocessing Variant Hierarchy

1. **Primary File Variant:** 200-region, band-pass filtered, no global signal regression, acquisition run 1 (`rest_1`).
   - *Example convention:* files matching `*rois_cc200*` from the filtered/no-GSR package.
2. **Fallback Variant Rule:** If the specific band-pass filtered, non-GSR 200-region variant does not exist in the distribution, the pipeline will utilize the **only available 200-region variant**.
3. **Exploratory Status of Other Variants:** All other atlas scales (e.g., AAL, CC400), global signal regressed (GSR) series, or alternative runs (`rest_2`) are designated strictly exploratory and will not be evaluated as primary endpoints.

---

## 4. Cohort Inclusion Rule, Sealing Procedure & Mechanical Recomputation

### 4.1. Comprehensive Cohort Inclusion Rule
* **Rule:** Include **every subject with a valid diagnostic label (TDC or ADHD) and a usable `rest_1` scan**, pooling across both the competition training cohort and the released-label test cohort if the Athena release contains them.
* **Excluded Categories:** Subjects with unreleased/missing diagnoses, uninterpretable scans, or scans failing file integrity checks. Single-class institutions (e.g., WashU and Pittsburgh, containing 100% controls and 0% ADHD in the competition release) are excluded a priori from classification folds.

### 4.2. Mechanical Recomputation Rule
* Preliminary counts ($N = 404$ development, $N = 216$ NYU sealed, $N = 620$ total) are cited from competition training tables and remain unverified until physical files are audited.
* **Mechanical Recomputation:** After physical archive inspection via `scripts/athena_inspect.py`, exact sample counts, the designated sealed institution, and the development cross-validation folds will be **recomputed mechanically strictly from the pre-registered written sealing rule**.
* *Protocol Clarification:* Mechanical recomputation of sample counts and folds based on physical file availability is a planned procedural step and is **explicitly not a protocol deviation**.
* **Audit Trail:** The final verified subject and site counts will be committed to a dated file (e.g., `docs/study2/COHORT_COUNTS_<DATE>.md`) prior to model training.

### 4.3. Written Rule for Sealing an Institution
To ensure an untainted external validation test, one institution must be completely sealed until all development models, motion thresholds, and baseline comparisons are finalized. The sealed institution is selected according to the following pre-registered rule:
> *The sealed institution must have balanced representation of both diagnostic classes (>5% ADHD), standard pediatric acquisition parameters, and minimal or zero exposure during historical pipeline exploration.*

* **Candidate Sealed Institution:** **NYU** ([UNVERIFIED] phenotypic count: $N = 216$: 98 TDC, 118 ADHD). Only 6 NYU subjects were included in the historical 153-subject C-PAC cohort.
* **Sealed-Site Execution Procedure:**
  1. The locked 4-step pipeline and Random Forest estimator are refitted **exactly once** on the pooled data of all development institutions combined.
  2. The frozen refitted model is scored **exactly once** on the sealed institution.
  3. **Nothing is re-selected, re-tuned, or modified afterwards.**

### 4.4. Leave-One-Institution-Out (LOIO) Development Cohort
Cross-validation will be conducted using Leave-One-Institution-Out (LOIO) across the remaining multi-class institutions:
* **KKI:** [UNVERIFIED] $N = 83$ (61 TDC, 22 ADHD)
* **NeuroIMAGE:** [UNVERIFIED] $N = 48$ (23 TDC, 25 ADHD)
* **OHSU:** [UNVERIFIED] $N = 79$ (42 TDC, 37 ADHD). *Note: 14 OHSU controls were present in historical C-PAC exploration.*
* **Peking:** [UNVERIFIED] $N = 194$ (116 TDC, 78 ADHD across Peking 1, 2, 3)

---

## 5. Quality Control, Motion Handling & ROI Rules

### 5.1. A Priori ROI Set Definition & Sealed Site Handling Rule
1. **Development ROI Set Definition:** The valid set of ROIs is determined **strictly from the four development institutions** (KKI, NeuroIMAGE, OHSU, Peking). An ROI must exhibit non-zero variance across **100% of analyzed subjects in the development cohort**. Any ROI that is constant or empty in even one development subject is dropped prior to computing connectivity edges.
2. **Pre-Specified Sealed Site Rule:** The sealed site cannot alter the trained feature space. If an ROI included in the development feature set happens to be missing or constant across one or more subjects in the sealed site:
   - That ROI's connectivity values for the affected subject(s) will be set to zero (representing prior mean correlation $r=0, z=0$).
   - The trained feature set, selected top-100 ANOVA edges, and model weights will **not** be re-selected or modified.

### 5.2. Motion File Verification Rule
Before computing framewise displacement, the exact column ordering and measurement units will be verified from one representative motion file header/documentation:
* Expected AFNI `3dvolreg` format: three rotational displacements in degrees ($roll, pitch, yaw$) followed by three translational displacements in millimeters ($dS, dL, dP$).
* *Protocol Clarification:* Verifying column order and unit scale prior to running numerical calculations is a standard data sanity verification and is **not a protocol deviation**.

### 5.3. Power's Framewise Displacement (FD) Formula & Primary Exclusion Threshold
Head motion will be quantified from the verified 6 rigid-body motion parameters using Power et al. (2012) Framewise Displacement:
1. Rotations are converted to arc-length displacements on the surface of a sphere of radius $r = 50\,\text{mm}$:
   $$d_{\theta, i} = \theta_i \times \frac{\pi}{180} \times 50\,\text{mm}$$
2. Frame-to-frame displacement for frame $i$ is calculated as:
   $$\text{FD}_i = |\Delta t_{x,i}| + |\Delta t_{y,i}| + |\Delta t_{z,i}| + |\Delta d_{roll,i}| + |\Delta d_{pitch,i}| + |\Delta d_{yaw,i}|$$
   where $\Delta x_i = x_i - x_{i-1}$ and $\text{FD}_0 = 0$.

* **Primary Exclusion Threshold:** A scan is excluded if:
  $$\text{Mean FD} > 0.50\,\text{mm} \quad \text{OR} \quad \text{Percentage of frames with } \text{FD} > 0.50\,\text{mm} > 20\%$$
* **Attrition Reporting Rule:** The number of excluded participants must be reported broken down by site and diagnosis (TDC vs. ADHD) to verify whether motion exclusion introduces differential attrition bias.

### 5.4. Exploratory Sensitivity Analyses for Motion
To evaluate the impact of motion filtering on model performance and class attrition, two pre-registered exploratory sensitivity analyses will be conducted:
1. **Stricter Motion Threshold:** $\text{Mean FD} > 0.20\,\text{mm}$ or $> 10\%$ of frames with $\text{FD} > 0.20\,\text{mm}$ excluded.
2. **No Motion Exclusion:** All participants with complete time series retained regardless of motion parameters.
* For both sensitivity analyses, exact participant counts excluded per site and per diagnosis (TDC vs. ADHD) will be tabulated alongside model ROC-AUC to assess stability.

---

## 6. Baselines & Benchmark Comparators

To prevent deceptive conclusions arising from confounding variables, the locked pipeline will be compared against two non-imaging baselines evaluated under the exact same LOIO folds:
1. **Motion-Only Baseline:** A logistic regression classifier trained exclusively on head motion metrics ($\text{Mean FD}$ and $\text{Max Displacement}$) with balanced class weighting.
2. **Age & Sex Baseline:** A logistic regression classifier trained on standardized age and biological sex:
   ```python
   StandardScaler() -> LogisticRegression(C=1.0, class_weight='balanced')
   ```

* **Definition of Outperforming the Motion Baseline:**
  The functional connectivity pipeline is defined as statistically outperforming the motion baseline if the **paired bootstrap difference in LOIO ROC-AUC** ($B = 2,000$ resamples) yields a **lower 95% confidence interval bound strictly greater than zero**:
  $$\Delta\text{AUC}_{\text{motion}} = \text{AUC}_{\text{fMRI}} - \text{AUC}_{\text{motion}}, \quad \text{CI}_{95\%}[\Delta\text{AUC}_{\text{motion}}] > 0$$

---

## 7. Statistical Analysis & Endpoints

### 7.1. Primary Endpoint: Pooled LOIO AUC
* **Primary Metric:** **Pooled Leave-One-Institution-Out (LOIO) ROC-AUC** computed across all pooled out-of-fold predictions on the development cohort ($N = 404$).
* **Secondary Breakdowns:** Per-institution ROC-AUCs (KKI, NeuroIMAGE, OHSU, Peking) will also be reported to characterize cross-scanner variability. However, the **pooled LOIO AUC is the single designated primary endpoint**.
* **Significance Testing:** Non-parametric permutation test ($B = 1,000$ iterations). Labels are permuted **strictly within each institution** to preserve site-specific base rates and demographic structures:
  $$p_{\text{perm}} = \frac{1 + \sum_{b=1}^B \mathbb{I}(\text{AUC}_b \ge \text{AUC}_{\text{observed}})}{1 + B}$$
* **Confidence Interval:** Stratified bootstrap 95% confidence interval ($B = 2,000$ resamples).

### 7.2. Secondary Endpoint & Statistical Power Caveat
* **Metric:** Incremental diagnostic value over non-imaging demographic covariates:
  $$\Delta\text{AUC} = \text{AUC}_{\text{fMRI}} - \text{AUC}_{\text{Age/Sex}}$$
  evaluated via paired stratified bootstrap.
* **Power Caveat:**
  > Assuming a true moderate increment of $\Delta\text{AUC} = \mathbf{+0.04}$, statistical power to reject $H_0: \Delta\text{AUC} \le 0$ at two-sided $\alpha = 0.05$ is only **about 18% at $N = 404$** and **about 26% at $N = 620$**, due to substantial covariance between fMRI connectivity predictions and demographic characteristics. A non-significant $\Delta\text{AUC}$ cannot be interpreted as proof that functional connectivity carries no signal, but rather reflects the sample size constraints of the dataset.

---

## 8. Success Criteria, Outcome Zones & Interpretation

### 8.1. Development Success Criteria
The locked pipeline demonstrates positive development signal if:
1. Primary pooled LOIO ROC-AUC significantly exceeds chance ($p_{\text{perm}} < 0.05$).
2. Primary pooled LOIO ROC-AUC statistically outperforms the motion-only baseline (paired bootstrap $\text{CI}_{95\%}[\Delta\text{AUC}_{\text{motion}}] > 0$).

### 8.2. Sealed Site Outcome Zones & Explicit Conclusions
When evaluated strictly once on the sealed institution (candidate: NYU, $N \approx 216$), results will fall into one of three pre-defined interpretive zones. The **95% bootstrap confidence interval lower bound governs** all definitive claims:

| Outcome Zone | Statistical Condition | Scientific Conclusion |
| :--- | :--- | :--- |
| **Confirmed Generalization (Success)** | 95% bootstrap CI lower bound $> 0.50$ | **Out-of-site generalization is confirmed.** Resting-state functional connectivity provides statistically reproducible predictive signal on a completely unseen, sealed institution without site-specific re-tuning. |
| **Inconclusive Zone** | Point estimate $> 0.50$, but 95% bootstrap CI lower bound $\le 0.50$ | **Results are statistically ambiguous.** While a positive discrimination point estimate is observed, the confidence interval spans chance ($0.50$). Due to the sample size constraint (the 95% CI half-width is approximately $\pm 0.07$ at $N \approx 216$), statistical power is insufficient to confirm discrimination, and external validity cannot be claimed. |
| **No Evidence of Transfer** | Point estimate $\le 0.50$ | **No evidence of out-of-site transfer.** The model performs at or below chance on the unseen clinical institution. Because the 95% CI half-width is about $\pm 0.07$ at $N = 216$, a point estimate at or below 0.50 does not strictly rule out a tiny population effect, but demonstrates that resting-state connectivity under this pipeline provides no usable diagnostic transfer. |

---

## 9. Atlas Divergence Acknowledgment

The Athena pipeline derivatives were generated via AFNI using an AFNI MNI template space, whereas Study 1 utilized C-PAC derivatives coregistered to the Craddock 2012 CC200 NIfTI template in FSL space. Due to potential discrepancies in coordinate grids, bounding boxes, and ROI indexing, **the Athena CC200 atlas cannot be assumed identical to the Study 1 CC200 atlas**. Study 2 is an independent multi-site evaluation, and quantitative results are not directly comparable to Study 1.
