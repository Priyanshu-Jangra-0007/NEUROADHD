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
4. **Primary Incremental Exploratory Contrast (Late Fusion):**
   $$\Delta \text{AUC}_{\text{incremental}} = \text{AUC}\Big(\text{Fused}[\text{Motion} + \text{Age} + \text{Sex},\ \text{Connectivity}]\Big) - \text{AUC}\Big(\text{Motion} + \text{Age} + \text{Sex}\Big)$$
   where the combined model is defined via **late fusion**:
   - Within each outer LOIO fold (holding out 1 institution $S_{\text{test}}$, training on 4 institutions $S_{\text{train}}$), **inner 4-fold LOIO** is executed strictly across the 4 training institutions (holding out each $S_{\text{in}} \in S_{\text{train}}$ and training on the remaining 3). No held-out institution $S_{\text{test}}$ ever touches the inner models or fusion weights.
   - Inner out-of-fold probability scores $s_{\text{cov}}^{\text{oof}}$ (covariate-only `LogisticRegression(C=1.0)`) and $s_{\text{conn}}^{\text{oof}}$ (the candidate connectivity configuration) are collected across all subjects in $S_{\text{train}}$.
   - A 2-parameter logistic regression meta-learner (`LogisticRegression(C=1.0, class_weight='balanced')`) is fit on $[s_{\text{cov}}^{\text{oof}}, s_{\text{conn}}^{\text{oof}}]$ against $y_{\text{train}}$.
   - Base models fit on all of $S_{\text{train}}$ predict $[s_{\text{cov}}^{\text{test}}, s_{\text{conn}}^{\text{test}}]$ on $S_{\text{test}}$, and the meta-learner outputs the final fused probability.
   - **Why Late Fusion Replaced Feature Concatenation (Decided Before Real Labels):** Under the earlier feature-concatenation design (`[cov, X_conn]`), concatenating 4 standardized covariates with 100 or 17,955 noisy connectivity edges caused severe feature dilution and overfitting (in `All Edges + RF`, Random Forest sampled $\sqrt{17959} \approx 134$ features per split and starved the 4 covariates, depressing null $\Delta\text{AUC}$ to $-0.18$; in `k100 + L2 LR`, noise edges depressed null $\Delta\text{AUC}$ to $-0.14$). Late fusion isolates the 2-parameter combination from high-dimensional edge noise so that pure-noise connectivity preserves the covariate-only AUC ($\Delta\text{AUC} \approx 0$).
5. **Pre-Declared Criteria for an "Interesting" Finding (Strict Conjunction on Primary Cohort):**
   - **Primary Cohort vs. Sensitivity Analyses:** The **PRIMARY analysis** for the "interesting" rule is the **$N = 564$ clean cohort** (NYU 196, Peking 189, NeuroIMAGE 42, KKI 79, OHSU 58). The $N = 620$ all-scans cohort, the **215-pair matched** subset, and the **226-pair matched** subset are **sensitivity analyses**—reported in full but not used to trigger the "interesting" rule.
   - **Chosen Configuration Selection Rule & Failure Policy:** The "chosen configuration" is selected as the configuration achieving the **largest pooled $\Delta\text{AUC}_{\text{incremental}}$** (fused minus covariate-only)—the exact same statistic used by the max-statistic permutation test. Any configuration that fails to converge or errors out is reported as failed and **excluded from the max statistic in the observed data AND in every permutation**.
   - **Strict 3-Part Conjunction:** To be declared **"interesting"**, the chosen configuration on the primary $N = 564$ cohort must satisfy **all three** conditions simultaneously:
     - **(a) Baseline Superiority with Bootstrap CI > 0:** Beats both the motion-only baseline and the age/sex baseline with 95% paired-bootstrap confidence interval lower bounds strictly above zero ($\text{CI}_{\text{lo}}(\text{AUC}_{\text{fused}} - \text{AUC}_{\text{motion}}) > 0$ **and** $\text{CI}_{\text{lo}}(\text{AUC}_{\text{fused}} - \text{AUC}_{\text{age/sex}}) > 0$). *(Note: These confidence intervals are conditional on configuration selection and are thus subject to selection optimism / winner's curse).*
     - **(b) Incremental Contrast Significance & Bootstrap CI > 0:** The pooled $\Delta\text{AUC}_{\text{incremental}}$ of the fused model over the covariate-only model (`Motion + Age + Sex`) is strictly $> 0$ with a 95% paired-bootstrap CI lower bound $> 0$ **AND** the max-statistic permutation $p < 0.05$ (repeating full selection over the 12 primary configurations).
     - **(c) Cross-Site Consistency:** Per-institution $\Delta\text{AUC}_{\text{incremental}} > 0$ in at least **4 of the 5** institutions.
   - **Pre-Declared Outcome Wording (Always Accompanied by Power Table):**
     - If all three conditions are met: *"a hypothesis for a new cohort; NYU is not sealed, so this is not confirmation"*.
     - Anything else: *"no evidence of incremental connectivity signal in this design"*.
     - Both outcomes must always be reported alongside the power table (full-rule pass rate $\approx 8\%$ at connectivity-only AUC $0.55$ and $\approx 22\%$ at $0.60$, 50 simulated datasets each, so the design cannot rule out small effects).
   - **Pre-Specified Random Seeds:**
     - Classifiers (`RandomForestClassifier`, `LogisticRegression`), bootstrap resamples (`DEFAULT_BOOTSTRAP_B = 2000`), and base permutation RNG: `DEFAULT_RANDOM_STATE = 42` (per-permutation classifier seed `42 + b`).
     - Synthetic power simulation seeds: `30000 + int(sig * 1000) + ds_idx` (`ds_idx in 0..49`, i.e. `30000–30049` for Null, `30240–30289` for Weak 1, `30280–30329` for Weak 2).
     - Motion matching: zero randomness (fully deterministic).
6. **Zero Individual Data Leakage:** Reports, tables, logs, and artifacts must never print individual subject IDs, raw timeseries, or per-subject prediction rows.
7. **Data Flow (Label-Free First, Labels Last), Pre-Flight Check, Null-Only Checkpointing, Write-Once Output, and Crash/Deviation Policy:**
   - **Strict Label-Free-First Data Flow:** `load_label_free_inputs()` first computes and caches all imaging features (CC200 Fisher-$z$ edges, timeseries for tangent space), motion covariates (saved dev FD + NYU FD from `intermediate_nyu.tar` via `compute_athena_fd.py`), age, and sex **without reading any label column**. `load_labels_last()` reads diagnosis (`DX`) **only** after all label-free inputs are built and only inside `--execute-real` after the `exploratory-frozen` guard passes.
   - **`rest_1`-Only Invariant for Primary & Sensitivity Cohorts:** `CFG_01`–`CFG_12` and the primary $N=564$ (plus sensitivity $N=620$, 215-pair, 226-pair) analyses use **`rest_1` ONLY** at every site. Multi-run averaging (`rest_1`/`rest_2`/`rest_3`) is confined strictly to the separate `CFG_13`–`CFG_16` family.
   - Before any real execution, `--preflight` verifies that all required inputs (features, covariates including `NYU_motion.csv` with 396 rows = 216 `rest_1` + 180 `rest_2` covering all 216 NYU `rest_1` scans, `intermediate_nyu.tar` with 2,665 entries = 1,368 regular files + 1,297 directories, scan lengths, and matching inputs) exist with expected shapes and counts **without reading any label column**.
   - **Null-Only Checkpointing:** During permutation testing, `--checkpoint` stores **only** the completed permutation null statistics (`test1_null_max_aucs`, `test2_null_max_deltas`)—never any observed AUCs, deltas, or partial results—allowing crash recovery under `caffeinate -dimsu` without exposing partial findings.
   - The results JSON file is strictly **write-once** (refuses to overwrite an existing file).
   - **Crash Policy:** If the run crashes, fix **only** the cause of the crash, record the deviation in a dated file (e.g. `docs/exploratory/DEVIATION_YYYY-MM-DD.md`), and **do not re-run after viewing partial output**.

---

## 2. Cohort Stratification & Cross-Validation Architecture

### 2.1 Included Multi-Class Institutions ($N = 620$ Scanned Subjects)
The 5 multi-class institutions with resting-state CC200 (190 populated ROIs) functional time courses and motion transform matrices:
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
- **Pittsburgh** ($N = 98$ phenotype rows, $N = 89$ archive subjects): 100% control in CC200 (190 populated ROIs) release (0 ADHD cases in imaging archive; single-class site).
- **WashU** ($N = 61$ phenotype rows, $N = 61$ archive subjects): 100% control in CC200 (190 populated ROIs) release (single-class site). All WashU runs (including 44 `rest_2` and 36 `rest_3`) are excluded from modeling.
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
   - Pairwise sample Pearson correlation matrix $R$ across CC200 (190 populated ROIs), clipped to $[-1 + 10^{-7}, 1 - 10^{-7}]$, Fisher-$z$ transformed ($z = \operatorname{arctanh}(r)$).
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

### 4.2 Limitation: Scan Length Heterogeneity & Rank Deficiency ($T < K = 190$ at KKI, NYU, and OHSU)
A critical neuroimaging limitation must be formally recognized across the measured `rest_1` scan lengths (`min / median / max` retained volumes):
- **KKI ($N = 83$):** `120 / 120 / 148` ($T < 190$ ROIs)
- **NYU ($N = 216$):** `172 / 172 / 172` ($T < 190$ ROIs)
- **OHSU ($N = 79$):** `72 / 74 / 74` ($T < 190$ ROIs)
- **Peking ($N = 194$):** `232 / 232 / 232` ($T > 190$ ROIs)
- **NeuroIMAGE ($N = 48$):** `257 / 257 / 257` ($T > 190$ ROIs)

Because $T < K = 190$ ROIs at **KKI, NYU, and OHSU**, empirical sample covariance matrices $S_i$ at these three institutions are **mathematically singular** ($\text{rank}(S_i) \le T - 1 < 190$). Tangent features at KKI, NYU, and OHSU therefore depend heavily on Ledoit-Wolf shrinkage regularization, whereas Peking and NeuroIMAGE ($T > 190$) do not. This creates a **site-dependent artifact** in covariance conditioning across LOIO folds; all tangent-space configurations (`CFG_05`–`CFG_08`) must be interpreted with this caveat.

---

### 4.3 The 4 Multi-Run Configurations (NYU + OHSU Separate Family)
Because only NYU ($180$ `rest_2` runs) and OHSU ($78$ `rest_2`, $78$ `rest_3` runs) acquired multiple resting-state sessions, multi-run aggregation (configurations `CFG_13`–`CFG_16`) is evaluated as a **separate family**:
- **Design:** Averaging Fisher-$z$ Pearson correlation matrices across all available runs of each subject, boosting temporal scan duration ($T_{\text{eff}} \approx 2\times\text{--}3\times$) and SNR. One row and one label per unique subject.
- **Subject-Grouped 2-Fold Cross-Validation:** Evaluated across **2 outer folds** (`NYU` vs. `OHSU`: train on one institution, test on the other). Because each outer fold trains on a single institution, late-fusion inner OOF scores use **5-fold subject-grouped `StratifiedKFold`** strictly within the training institution. Effective sample size $N$ remains the number of unique subjects.
- **Separate Multiple-Testing Correction:** Because `CFG_13`–`CFG_16` are evaluated on NYU + OHSU only and are **not comparable across the 5 sites** (KKI, NeuroIMAGE, and Peking have only single runs; WashU runs are excluded as single-class), this family uses **its own Bonferroni correction across its 4 tests** ($\alpha_{\text{adj}} = 0.05 / 4 = 0.0125$).

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
- Labels are permuted strictly **within each institution** (as in Study 2), preserving per-site diagnostic prevalence.
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

## 6. Pre-Declared Permutation Scope, Total Runtime & >4-Hour Policy (Estimated)

- **Pre-Declared Permutation Scope:**
  - **Primary $N = 564$:** Test 1 (within-site label permutation) and Test 2 (within-site connectivity-row permutation, late fusion) run **200 permutations each**.
  - **Sensitivity Analyses ($N = 620$, 215-pair matched, 226-pair matched):** Report observed AUCs, $\Delta\text{AUC}$s, and 95% paired/stratified bootstrap CIs **with NO permutation $p$-values**.
  - **`CFG_13`–`CFG_16` Separate Family (NYU + OHSU):** Reports **200-permutation** Bonferroni-corrected $p$-values across its 4 tests ($\alpha_{\text{adj}} = 0.0125$).

*(All timings below are labeled as **estimates**, measured on synthetic data at exact real site sizes with `C = 1.0` and late fusion)*:

| Component of Full `--execute-real` Invocation | Scope | Estimated Time |
| :--- | :--- | :--- |
| **Label-Free Archive Build (`load_label_free_inputs`)** | 620 `rest_1` + 336 `rest_2/3` + NYU FD | $\approx 26.5\text{ s}$ (`0.5 s` if cached) |
| **Primary $N = 564$ Observed + Sensitivity Observed (`N=620`, `215p`, `226p`)** | 4 cohorts $\times$ 12 configs (15-split tangent cache) | $\approx 3.0\text{ min}$ |
| **Primary $N = 564$ Test 1 + Test 2 (200 Perms Each)** | T1: $5.6\text{ min}$; T2: $88.4\text{ min}$ | $\approx 93.9\text{ min}$ |
| **`CFG_13`–`CFG_16` Family (NYU+OHSU, 4 Configs $\times$ 200 Perms)** | 2-fold LOIO + 5-fold inner CV | $\approx 15.0\text{ min}$ |
| **Total Entire `--execute-real` Invocation** | **All analyses combined** | **$\approx 112\text{ min}$ ($\approx 1.9\text{ hr}$)** |

- **Policy if Total Runtime Exceeds 4 Hours:** Execution runs under `caffeinate -dimsu` with null-only checkpointing (`--checkpoint`). If total runtime exceeds 4 hours, execution is halted safely without viewing partial AUCs, the exact cause/stage is recorded in a dated deviation file (`docs/exploratory/DEVIATION_YYYY-MM-DD.md`), and remaining permutations are resumed strictly from the null-only checkpoint (never re-run after viewing partial output).

---

## 7. Honest Synthetic Power Simulation (Estimated, Late Fusion)

To estimate both the **false-positive rate** and **statistical power** of the full 3-part "interesting" conjunction under realistic weak-effect regimes at the exact real site sizes ($N = 564$: NYU 196, Peking 189, NeuroIMAGE 42, KKI 79, OHSU 58; $p = 17,955$ edges; fixed seeds; covariates around $\text{AUC} \approx 0.60$):
- **Simulation Design:** **50 simulated datasets** per regime (reduced from 100 due to compute cost—814 s total—never below 50) and **99 within-site connectivity-row permutations** per dataset with **no early stopping** (so $p = (1 + \text{exceed}) / 100$).
- **Why Earlier 100% Figures Were Invalid:** The earlier power table reported $100\%$ power at $\Delta\text{AUC} \approx 0.03\text{--}0.05$ because (1) it used an overly strong planted signal calibrated under the flawed feature-concatenation design, and (2) under concatenation, noise-edge overfitting depressed the null $\Delta\text{AUC}$ to $-0.139$, causing any modest positive gain to separate trivially from a depressed null. Those earlier figures are discarded.

| Signal Regime (Estimated) | Mean Covariate AUC | Mean Conn-Only AUC | Mean Fused AUC | Mean $\Delta\text{AUC}$ | Perm $p < 0.05$ Alone (95% Wilson CI) | **Full 3-Part Conjunction Rate (95% Wilson CI, Estimate)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Null ($\text{conn} \approx 0.50$)** | $0.603$ | $0.502$ | $0.583$ | $-0.0201$ | $5/50$ ($10.0\%$, $[4.3\%, 21.4\%]$) | **$0/50$ ($0.0\%$, $[0.0\%, 7.1\%]$)** *(False-Positive Rate)* |
| **Weak 1 ($\text{conn} \approx 0.55$)** | $0.604$ | $0.569$ | $0.603$ | $-0.0008$ | $20/50$ ($40.0\%$, $[27.6\%, 53.8\%]$) | **$4/50$ ($8.0\%$, $[3.2\%, 18.8\%]$)** *(Power)* |
| **Weak 2 ($\text{conn} \approx 0.60$)** | $0.609$ | $0.613$ | $0.635$ | $+0.0263$ | $40/50$ ($80.0\%$, $[67.0\%, 88.8\%]$) | **$11/50$ ($22.0\%$, $[12.8\%, 35.2\%]$)** *(Power)* |

*(All values labeled as estimates).* Under late fusion, the full conjunction has a $0.0\%$ false-positive rate under the null and $8.0\%\text{--}22.0\%$ power at weak connectivity effects ($\text{AUC} \approx 0.55\text{--}0.60$).
