# Study 2 Report: Preregistered Multi-Site Generalization of a Locked Resting-State fMRI Pipeline on ADHD-200 Athena Derivatives

**Headline Result:** Development success criteria (`PREREGISTRATION.md`, Section 8.1) were **not met** in either the primary motion-filtered analysis ($N = 368$) or the registered Section 5.4 no-exclusion sensitivity analysis ($N = 404$), and the one-shot sealed NYU evaluation falls in the registered **Inconclusive Zone** (primary $N = 196$: ROC-AUC `0.5765`, 95% CI `[0.496177, 0.656518]`; sensitivity $N = 216$: ROC-AUC `0.5770`, 95% CI `[0.499996, 0.652819]`). Simple motion-only and age/sex baselines scored as high or higher than the locked fMRI pipeline in every regime.

**Date:** 2026-11-18  
**Branch:** `feat/study2-athena-acquisition`  
**OSF Preregistration (Primary):** [https://osf.io/grbt2](https://osf.io/grbt2) (DOI: `10.17605/OSF.IO/GRBT2`)  
**OSF Registration (Open-Ended, Registered First):** [https://osf.io/db7nv](https://osf.io/db7nv) (DOI: `10.17605/OSF.IO/DB7NV`)  
**Locked Protocol Commit & SHA-256:** `2d7ed3b3be3780c0b8a5b3de7d3c786bb77df19e` (`2d7ed3b`), `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec` ([`docs/study2/PREREGISTRATION.md`](PREREGISTRATION.md), unmodified)

---

## Abstract

Study 2 evaluated whether a locked resting-state functional connectivity pipeline from Study 1 generalizes across ADHD-200 institutions on Athena CC200 derivatives without data leakage, and whether connectivity outperforms head-motion and age/sex baselines. The protocol was preregistered on OSF (`https://osf.io/grbt2`, commit `2d7ed3b`) before acquiring or inspecting any Athena data. Leave-one-institution-out (LOIO) cross-validation was run across four development institutions (KKI, NeuroIMAGE, OHSU, Peking; primary motion-filtered $N = 368$, registered Section 5.4 no-exclusion sensitivity $N = 404$), followed by a single locked evaluation on the sealed NYU site (primary $N = 196$, sensitivity $N = 216$). Development Section 8.1 success criteria were not met in either the primary analysis or the registered Section 5.4 sensitivity analysis. On sealed NYU, both the governing primary evaluation and the no-exclusion sensitivity evaluation (95% bootstrap CI lower bound `0.499996`) fell in the registered Inconclusive Zone. Motion-only and age/sex baselines scored as high or higher than the fMRI pipeline across all four regimes. External validity cannot be claimed.

---

## 1. Introduction

In Study 1, an exploratory benchmark on 153 C-PAC preprocessed ADHD-200 subjects produced a locked resting-state functional connectivity pipeline combining Fisher $z$-transformed pairwise Pearson correlations, in-fold median imputation, zero-variance filtering, standardization, top-$100$ ANOVA $F$-score selection, and a shallow balanced Random Forest classifier (`n_estimators=50`, `max_depth=3`, `class_weight='balanced'`, `random_state=42`; [`PREREGISTRATION.md`](PREREGISTRATION.md), Sections 1–2).

The objective of Study 2 — a **pre-registered follow-up on a different preprocessing pipeline** — was to evaluate whether that locked Study 1 pipeline generalizes across a larger multi-site ADHD-200 cohort preprocessed with the Athena pipeline without data leakage, and whether resting-state functional connectivity provides diagnostic classification beyond head motion and demographic covariates (age and biological sex; [`PREREGISTRATION.md`](PREREGISTRATION.md), Section 1). Because Study 2 differs from Study 1 in preprocessing pipeline (Athena vs. C-PAC), atlas representation (`190` populated ROIs vs. `200`), cohort composition, and sites (with `126` development subjects and `6` NYU subjects overlapping Study 1), Study 1's exploratory `0.62` LOIO AUC not holding up in this follow-up is not a direct test of the same protocol. This conclusion applies specifically to this locked pipeline, not to functional connectivity in general. Prior to downloading, inspecting, or analyzing any Athena derivative files, the protocol was committed (`2d7ed3b`) and registered publicly on OSF.

---

## 2. Prior Exposure Disclosure

Before interpreting any Study 2 results, prior subject overlap with Study 1 must be stated prominently (documented in full in [`OSF_RECORD.md`](OSF_RECORD.md) and [`OSF_REGISTRATION.md`](OSF_REGISTRATION.md)):

1. **126 Development Subjects and 6 NYU Subjects Overlapped with Study 1:** Out of the 153-subject Study 1 exploratory C-PAC cohort where the locked pipeline architecture and hyperparameters were selected, **126 subjects** also appear in the Study 2 development cohort (**KKI:** `24`, **NeuroIMAGE:** `19`, **Peking 1–3:** `69`, **OHSU:** `14` controls), and **6 subjects** appear in the sealed NYU cohort.
2. **OSF Foreknowledge Option Understates Actual Overlap:** The OSF registration form selected the prior-observation statement *"Authors' limited observation of the data could not influence their analysis decisions."* As corrected in [`OSF_RECORD.md`](OSF_RECORD.md), that option **understates** the actual prior exposure: because those 126 development subjects (and 6 NYU subjects) were part of Study 1 where the locked pipeline hyperparameters were chosen (under C-PAC rather than Athena), Study 2's development evaluation is not strictly naive to those 126 individuals.
3. **Zero Athena Exposure at Registration:** At the time of commit `2d7ed3b` and both 2026-11-17 OSF registrations — `db7nv` at `14:56 UTC` and `grbt2` at approximately 14:58 UTC (inferred from the OSF page display) — no Athena `.1D` BOLD time series, motion files, or connectivity matrices had been downloaded, unpacked, inspected, or analyzed.

---

## 3. Methods

### 3.1. Data Source and Preprocessing Variant
Precomputed Athena derivatives from the ADHD-200 Consortium (Neuro Bureau NITRC Release 1600) were used under the registered hierarchy ([`PREREGISTRATION.md`](PREREGISTRATION.md), Section 3): 200-region Craddock parcellation (`*rois_cc200*` / `*cc200_TCs.1D`), band-pass filtered, without global signal regression (`ADHD200_CC200_TCs_filtfix.tar.gz`), acquisition run 1 (`rest_1`). Other atlases, GSR variants, and `rest_2` runs were excluded from primary evaluation.

### 3.2. Cohort Inclusion, Mechanical Recomputation, and Sealed Institution Rule
Per [`PREREGISTRATION.md`](PREREGISTRATION.md) Sections 4.1–4.4:
- **Inclusion Rule:** Every subject with a valid binary diagnostic label (`DX = 0` mapped to Typically Developing Control [TDC]; `DX ∈ {1, 2, 3}` mapped to ADHD) and a usable `rest_1` scan was included. Single-class or pending-label institutions (WashU: 61 TDC / 0 ADHD; Pittsburgh: 94 TDC / 4 ADHD in phenotype table, 0 ADHD in training archive; Brown: 26 pending) were excluded a priori.
- **Mechanical Recomputation:** Archive member audit matched **404 / 404** development `rest_1` scans across four multi-class institutions (**KKI** $N = 83$, **NeuroIMAGE** $N = 48$, **OHSU** $N = 79$, **Peking 1–3 combined** $N = 194$) with zero missing diagnoses ([`COHORT_COUNTS_2026-11-18.md`](COHORT_COUNTS_2026-11-18.md)).
- **Sealed Institution Selection:** Applying the written sealing rule (Section 4.3: balanced diagnostic classes `>5%` ADHD, standard pediatric acquisition, minimal Study 1 exposure) confirmed **NYU** as the sealed external site. Joining the `ADHD200_CC200_TCs_filtfix.tar.gz` member names (without opening imaging files) to `adhd200_preprocessed_phenotypics.tsv` yielded exactly **216** complete `rest_1` scans (**98 TDC, 118 ADHD**, `54.6%` ADHD). The `263` NYU rows in the consolidated phenotype TSV include `47` test-release rows absent from the CC200 archive and were not scored.

### 3.3. ROI Set Definition and Locked Model Pipeline
Per [`PREREGISTRATION.md`](PREREGISTRATION.md) Sections 2 and 5.1:
- **ROI Set:** Valid ROIs were defined strictly on the development cohort (`404` scans; requiring finite, non-zero temporal variance across `100%` of development scans). All `190 / 190` populated ROIs in `templates/ADHD200_parcellate_200.nii.gz` satisfied this rule (`0` dropped; minimum temporal `std = 0.004031`), yielding $P = 190 \times 189 / 2 = 17,955$ upper-triangle Fisher $z$-transformed Pearson correlation edges (off-diagonal correlations clipped to `[-1 + 1e-7, 1 - 1e-7]` prior to $\operatorname{arctanh}$).
- **Locked Pipeline (Refitted Inside Every Training Fold):**
  1. `SimpleImputer(strategy='median')`
  2. `VarianceThreshold(threshold=0.0)`
  3. `StandardScaler()`
  4. `SelectKBest(score_func=f_classif, k=100)`
  5. `RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42)`
- No hyperparameter search, threshold tuning, or feature re-selection was performed.

### 3.4. Head Motion, Framewise Displacement (FD), and Exclusion Regimes
Per [`PREREGISTRATION.md`](PREREGISTRATION.md) Sections 5.2–5.4 and [`DEVIATIONS.md`](DEVIATIONS.md) Branch (b):
- Because the six development `<Site>_preproc_xfms.tar` archives contained `0 / 404` AFNI `3dvolreg` `rp_*.1D` traces, Power et al. (2012) FD ($r = 50\text{ mm}$ sphere, $\text{FD}_0 = 0$) was computed from Neuro Bureau NIAK $4 \times 4 \times T_{\text{NIAK}}$ rigid-body transformation matrices conjugated to each scan's NIfTI volume geometric center with the **first 3 frames removed** (`DUMMY_FRAMES_DROPPED = 3` in `scripts/compute_athena_fd.py` and `scripts/run_study2_sealed_site.py`, retaining $T_{\text{NIAK}} - 3$ motion frames). Each Athena CC200 `.1D` file contains **$T_{\text{BOLD}} = T_{\text{NIAK}} - 4$ BOLD data rows** (matching Athena's 4-volume removal `3dcalc '[4..$]'`) plus **1 header row** ($T_{\text{BOLD}} + 1 = T_{\text{NIAK}} - 3$ total non-empty lines across all 404 development scans), so the frozen FD calculator dropped 3 leading NIAK frames rather than 4, retaining 1 extra leading NIAK frame (with $\text{FD}_0 = 0.0$) before the $T_{\text{BOLD}}$ BOLD data rows (see Section 5 and [`DEVIATIONS.md`](DEVIATIONS.md) Section 4.5).
- **Primary Exclusion Threshold (Section 5.3):** Exclude a scan if $\text{Mean FD} > 0.50\text{ mm}$ OR $>20\%$ of frames have $\text{FD} > 0.50\text{ mm}$.
- **Registered No-Exclusion Sensitivity Analysis (Section 5.4):** Retain all complete `rest_1` scans regardless of FD ($N = 404$ development, $N = 216$ NYU).
- **Strict Motion Threshold (Section 5.4):** $\text{Mean FD} > 0.20\text{ mm}$ OR $>10\%$ of frames with $\text{FD} > 0.20\text{ mm}$; retained for descriptive attrition reporting only (Section 5.1.4).
- Attrition by site and diagnosis was evaluated with two-sided Fisher exact tests.

### 3.5. Validation Design, Non-Imaging Baselines, and Inference Criteria
Per [`PREREGISTRATION.md`](PREREGISTRATION.md) Sections 4.3, 6, 7, and 8 (frozen in [`ANALYSIS_FREEZE_2026-11-18.md`](ANALYSIS_FREEZE_2026-11-18.md) and [`SEALED_FREEZE_2026-11-18.md`](SEALED_FREEZE_2026-11-18.md)):
- **Development Cross-Validation:** Leave-One-Institution-Out (LOIO) across KKI, NeuroIMAGE, OHSU, and Peking (1–3 combined). The single primary endpoint is pooled out-of-fold LOIO ROC-AUC (per-institution AUCs reported as secondary breakdowns).
- **Non-Imaging Baselines (Exact Same LOIO Folds):**
  1. **Motion-Only Baseline:** `StandardScaler()` → `LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000)` on `mean_fd_mm` and `max_disp_mm` (computed from the same NIAK volume-centered motion source).
  2. **Age & Sex Baseline:** Identical standardized balanced logistic regression on age and biological sex.
- **Inference Parameters (Locked Seeds = `42`):**
  - **H1 (Chance Discrimination):** Within-institution non-parametric permutation test ($B = 1,000$ permutations preserving site class rates), $p_{\text{perm}} = (1 + \sum \mathbb{I}(\text{AUC}_b \ge \text{AUC}_{\text{obs}})) / (1 + B)$, plus stratified bootstrap 95% CI ($B = 2,000$).
  - **H2 (Outperforming Motion Baseline):** Paired stratified bootstrap ($B = 2,000$) for $\Delta\text{AUC}_{\text{motion}} = \text{AUC}_{\text{fMRI}} - \text{AUC}_{\text{motion}}$; outperformance requires 95% CI lower bound strictly $> 0$.
  - **H3 (Incremental Value Over Age & Sex):** Paired stratified bootstrap ($B = 2,000$) for $\Delta\text{AUC}_{\text{age/sex}} = \text{AUC}_{\text{fMRI}} - \text{AUC}_{\text{Age/Sex}}$ (secondary endpoint; Section 7.2).
  - **Section 8.1 Development Success Criteria:** Met if and only if (1) primary pooled LOIO ROC-AUC satisfies $p_{\text{perm}} < 0.05$ **and** (2) paired bootstrap $\text{CI}_{95\%,\text{lo}}[\Delta\text{AUC}_{\text{motion}}] > 0$.
- **Sealed External Validation (H4; Section 4.3 & Section 8.2):** The locked pipeline was refitted once on pooled development data (`primary_368` for primary sealed evaluation; `no_exclusion_404` for sensitivity sealed evaluation) and scored once on NYU (`N = 196` primary; `N = 216` sensitivity) with stratified bootstrap 95% CIs ($B = 2,000$, seed `42`). Under Section 5.1, any NYU ROI that was constant or missing would be zero-filled without altering the trained feature set or weights; `0` NYU scans required zero-filling. Motion-only and age/sex baselines were also scored on NYU for descriptive comparison. Section 8.2 outcome zones are governed mechanically by the primary sealed 95% bootstrap CI lower bound:
  - **Confirmed Generalization (Success):** 95% bootstrap CI lower bound $> 0.50$.
  - **Inconclusive Zone:** Point estimate $> 0.50$, 95% bootstrap CI lower bound $\le 0.50$.
  - **No Evidence of Transfer:** Point estimate $\le 0.50$.

---

## 4. Results

### 4.1. Cohort Counts and Motion Attrition
All counts below are taken verbatim from [`COHORT_COUNTS_2026-11-18.md`](COHORT_COUNTS_2026-11-18.md) and [`SEALED_RESULTS_2026-11-18.md`](SEALED_RESULTS_2026-11-18.md).

#### Table 1. Development Primary Motion Exclusion (`>0.50 mm` OR `>20%` Frames) & Attrition Tests

| Site / Institution | Retained Total | Retained TDC | Retained ADHD | Excluded Total | Excluded TDC ($n$ / %) | Excluded ADHD ($n$ / %) | Fisher Exact $p$ (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KKI** | 79 | 60 | 19 | 4 | 1 (1.6%) | 3 (13.6%) | `0.0551` |
| **NeuroIMAGE** | 42 | 23 | 19 | 6 | 0 (0.0%) | 6 (24.0%) | `0.0227` |
| **OHSU** | 58 | 33 | 25 | 21 | 9 (21.4%) | 12 (32.4%) | `0.3140` |
| **Peking (Combined)** | 189 | 114 | 75 | 5 | 2 (1.7%) | 3 (3.8%) | `0.3930` |
| **POOLED DEV (`primary_368`)** | **368** | **230** | **138** | **36** | **12 (5.0%)** | **24 (14.8%)** | **`0.0011`** |

- Under the **no-exclusion sensitivity regime** (`no_exclusion_404`), all **404** development scans were retained (`242` TDC, `162` ADHD; KKI `83` [61/22], NeuroIMAGE `48` [23/25], OHSU `79` [42/37], Peking `194` [116/78]).
- Under the **strict threshold** (`>0.20 mm` OR `>10%` frames, descriptive only), **271 / 404** development scans were excluded (`144 / 242` [59.5%] TDC vs. `127 / 162` [78.4%] ADHD, Fisher exact $p = 0.0001$), leaving `133` scans (`98` TDC, `35` ADHD) and reducing 3 of 4 LOIO institution folds below 15 per class (`KKI` ADHD = `2`; `NeuroIMAGE` TDC = `11`, ADHD = `8`; `OHSU` TDC = `13`, ADHD = `6`).
- On the **sealed NYU site**, **216** complete `rest_1` scans (`98` TDC, `118` ADHD) were audited; **20** scans (`7` TDC [7.1%], `13` ADHD [11.0%], two-sided Fisher exact $p = 0.3568$) were excluded by the Section 5.3 primary FD rule, leaving **196** scans (`91` TDC, `105` ADHD) for `primary_sealed`. Zero NYU scans required Section 5.1 ROI zero-filling.

---

### 4.2. Development Primary Analysis — `primary_368` ($N = 368$, Registered Section 5.3 Exclusion)

Results from [`DEV_RESULTS_2026-11-18.md`](DEV_RESULTS_2026-11-18.md):

#### Table 2. Development Primary Pooled LOIO Endpoints ($N = 368$, $B_{\text{boot}} = 2,000$, $B_{\text{perm}} = 1,000$)

| Metric / Endpoint | Point Estimate | 95% Stratified Bootstrap CI | Permutation $p$ / Criterion Status |
| :--- | :---: | :---: | :---: |
| **H1: fMRI Pooled LOIO ROC-AUC** | **0.5241** (`0.524134`) | **`[0.4691, 0.5797]`** | `p_perm = 0.2298` (`229 / 1000` exceed; null mean `0.4924`, SD `0.0399`) |
| **Motion-Only Baseline Pooled LOIO AUC** | **0.6038** (`0.603812`) | `[0.5508, 0.6585]` | — |
| **Age & Sex Baseline Pooled LOIO AUC** | **0.6163** (`0.616273`) | `[0.5650, 0.6672]` | — |
| **H2: $\Delta\text{AUC}_{\text{motion}}$ ($\text{fMRI} - \text{Motion}$)** | **-0.0797** (`-0.079679`) | **`[-0.1520, -0.0099]`** | $\text{CI}_{\text{lo}} > 0$: **False** |
| **H3: $\Delta\text{AUC}_{\text{age/sex}}$ ($\text{fMRI} - \text{Age/Sex}$)** | **-0.0921** (`-0.092139`) | **`[-0.1649, -0.0169]`** | $\text{CI}_{\text{lo}} > 0$: **False** |

#### Table 3. Development Primary Per-Institution LOIO ROC-AUCs ($N = 368$)

| Institution | fMRI LOIO AUC | 95% Stratified Bootstrap CI | Motion Baseline AUC | Age/Sex Baseline AUC |
| :--- | :---: | :---: | :---: | :---: |
| **KKI** ($n = 79$) | **0.5500** (`0.550000`) | `[0.3930, 0.7026]` | `0.5825` | `0.4939` |
| **NeuroIMAGE** ($n = 42$) | **0.5698** (`0.569794`) | `[0.3959, 0.7529]` | `0.6018` | `0.6934` |
| **OHSU** ($n = 58$) | **0.6048** (`0.604848`) | `[0.4521, 0.7430]` | `0.6315` | `0.6279` |
| **Peking** ($n = 189$) | **0.5346** (`0.534620`) | `[0.4537, 0.6152]` | `0.6230` | `0.6730` |

**Section 8.1 Reading (`primary_368`):** Criterion 1 ($p_{\text{perm}} < 0.05$) was **not met** (`0.2298 >= 0.05`). Criterion 2 ($\text{CI}_{95\%,\text{lo}}[\Delta\text{AUC}_{\text{motion}}] > 0$) was **not met** (`-0.1520 <= 0`). Both the motion-only baseline (`0.6038`) and the age/sex baseline (`0.6163`) scored higher than the fMRI pipeline (`0.5241`). Combined Section 8.1 criteria were **not met**.

---

### 4.3. Development Registered Section 5.4 Sensitivity Analysis — `no_exclusion_404` ($N = 404$)

Results from [`DEV_RESULTS_2026-11-18.md`](DEV_RESULTS_2026-11-18.md) for the registered Section 5.4 no-exclusion sensitivity analysis:

#### Table 4. Development Registered Section 5.4 Sensitivity Pooled LOIO Endpoints ($N = 404$, $B_{\text{boot}} = 2,000$, $B_{\text{perm}} = 1,000$)

| Metric / Endpoint | Point Estimate | 95% Stratified Bootstrap CI | Permutation $p$ / Criterion Status |
| :--- | :---: | :---: | :---: |
| **H1: fMRI Pooled LOIO ROC-AUC** | **0.5740** (`0.574023`) | **`[0.5243, 0.6243]`** | `p_perm = 0.0050` (`4 / 1000` exceed; null mean `0.4830`, SD `0.0359`) |
| **Motion-Only Baseline Pooled LOIO AUC** | **0.6158** (`0.615805`) | `[0.5670, 0.6644]` | — |
| **Age & Sex Baseline Pooled LOIO AUC** | **0.5857** (`0.585667`) | `[0.5369, 0.6363]` | — |
| **H2: $\Delta\text{AUC}_{\text{motion}}$ ($\text{fMRI} - \text{Motion}$)** | **-0.0418** (`-0.041781`) | **`[-0.0890, +0.0087]`** | $\text{CI}_{\text{lo}} > 0$: **False** |
| **H3: $\Delta\text{AUC}_{\text{age/sex}}$ ($\text{fMRI} - \text{Age/Sex}$)** | **-0.0116** (`-0.011644`) | **`[-0.0795, +0.0568]`** | $\text{CI}_{\text{lo}} > 0$: **False** |

#### Table 5. Development Registered Section 5.4 Sensitivity Per-Institution LOIO ROC-AUCs ($N = 404$)

| Institution | fMRI LOIO AUC | 95% Stratified Bootstrap CI | Motion Baseline AUC | Age/Sex Baseline AUC |
| :--- | :---: | :---: | :---: | :---: |
| **KKI** ($n = 83$) | **0.4694** (`0.469449`) | `[0.3353, 0.6088]` | `0.6118` | `0.4832` |
| **NeuroIMAGE** ($n = 48$) | **0.5757** (`0.575652`) | `[0.4243, 0.7322]` | `0.7061` | `0.6348` |
| **OHSU** ($n = 79$) | **0.6042** (`0.604247`) | `[0.4755, 0.7239]` | `0.6461` | `0.6338` |
| **Peking** ($n = 194$) | **0.6732** (`0.673187`) | `[0.5985, 0.7477]` | `0.6112` | `0.6755` |

**Section 8.1 Reading (`no_exclusion_404`):** In this registered Section 5.4 sensitivity analysis, Criterion 1 was met (`p_perm = 0.0050 < 0.05`), while Criterion 2 was **not met** (`-0.0890 <= 0`). Both the motion-only baseline (`0.6158`) and the age/sex baseline (`0.5857`) scored higher than the fMRI pipeline (`0.5740`). Combined Section 8.1 criteria were **not met**.

---

### 4.4. Sealed External Validation Results — NYU (`primary_sealed` $N = 196$ and `sensitivity_no_exclusion` $N = 216$)

Results from [`SEALED_RESULTS_2026-11-18.md`](SEALED_RESULTS_2026-11-18.md) (single one-shot execution; nothing re-tuned or re-run):

#### Table 6. Sealed Primary Analysis — `primary_sealed` ($N = 196$: 91 TDC / 105 ADHD; Governs Section 8.2)

| Model / Endpoint | NYU ROC-AUC | 95% Stratified Bootstrap CI ($B = 2,000$, seed `42`) | Section 8.2 Outcome Zone |
| :--- | :---: | :---: | :---: |
| **Locked fMRI Pipeline (Primary — Governing)** | **0.5765** (`0.576452`) | **`[0.496177, 0.656518]`** | **Inconclusive Zone** (`inconclusive`) |
| **Motion-Only Baseline *(descriptive only)*** | **0.6069** (`0.606907`) | `[0.527556, 0.680484]` | — |
| **Age & Sex Baseline *(descriptive only)*** | **0.5806** (`0.580638`) | `[0.500044, 0.659267]` | — |

#### Table 7. Sealed Registered Section 5.4 Sensitivity Analysis — `sensitivity_no_exclusion` ($N = 216$: 98 TDC / 118 ADHD)

| Model / Endpoint | NYU ROC-AUC | 95% Stratified Bootstrap CI ($B = 2,000$, seed `42`) | Section 8.2 Outcome Zone |
| :--- | :---: | :---: | :---: |
| **Locked fMRI Pipeline (Sensitivity)** | **0.5770** (`0.576963`) | **`[0.499996, 0.652819]`** | **Inconclusive Zone** (`inconclusive`) |
| **Motion-Only Baseline *(descriptive only)*** | **0.6076** (`0.607575`) | `[0.533531, 0.682467]` | — |
| **Age & Sex Baseline *(descriptive only)*** | **0.6150** (`0.615012`) | `[0.536651, 0.692775]` | — |

**Section 8.2 Outcome Zone Reading:** In the governing primary sealed evaluation ($N = 196$), the fMRI point estimate (`0.576452`) exceeds `0.50` while the 95% bootstrap CI lower bound (`0.496177`) is $\le 0.50$. In the sealed sensitivity evaluation ($N = 216$), the point estimate (`0.576963`) exceeds `0.50` while the unrounded 95% bootstrap CI lower bound (`0.499996`) is $\le 0.50$. Both motion-only (`0.6069` at $N = 196$; `0.6076` at $N = 216$) and age/sex (`0.5806` at $N = 196$; `0.6150` at $N = 216$) descriptive baselines scored higher than the fMRI pipeline. Both sealed regimes fall in the registered **Inconclusive Zone**, quoted verbatim from [`PREREGISTRATION.md`](PREREGISTRATION.md#L159):

> | **Inconclusive Zone** | Point estimate $> 0.50$, but 95% bootstrap CI lower bound $\le 0.50$ | **Results are statistically ambiguous.** While a positive discrimination point estimate is observed, the confidence interval spans chance ($0.50$). Due to the sample size constraint (the 95% CI half-width is approximately $\pm 0.07$ at $N \approx 216$), statistical power is insufficient to confirm discrimination, and external validity cannot be claimed. |

---

## 5. Deviations and Clarifications

Every item logged in [`docs/study2/DEVIATIONS.md`](DEVIATIONS.md) is listed plainly below (`PREREGISTRATION.md` was never modified):

### 5.1. Protocol Deviations
1. **NIAK Rigid-Body Transforms Substituted for AFNI `3dvolreg` `rp_*.1D` (Registration Section 5.2) & Frame-Drop Clarification:** The six development `<Site>_preproc_xfms.tar` archives contained FSL FNIRT structural-to-template warp fields only (`0 / 404` AFNI `rp_*.1D` traces; commit `08f87fe`). Under the pre-inspection fallback rule committed prior to opening those archives (`DEVIATIONS.md`, Branch b, commit `ac517ce`), Power et al. (2012) FD ($r = 50\text{ mm}$, $\text{FD}_0 = 0$) was computed from Neuro Bureau NIAK $4 \times 4 \times T_{\text{NIAK}}$ rigid-body transformation matrices with the **first 3 frames removed** (`DUMMY_FRAMES_DROPPED = 3` in `scripts/compute_athena_fd.py` and `scripts/run_study2_sealed_site.py`, retaining $T_{\text{NIAK}} - 3$ motion frames). As documented in [`DEVIATIONS.md`](DEVIATIONS.md) Section 4.5, each Athena CC200 `.1D` file contains **$T_{\text{BOLD}} = T_{\text{NIAK}} - 4$ BOLD data rows** (Athena's 4-volume removal `3dcalc '[4..$]'`) plus **1 header row** ($T_{\text{BOLD}} + 1 = T_{\text{NIAK}} - 3$ total non-empty lines across all 404 development scans), so the frozen FD calculator dropped 3 leading NIAK frames rather than 4 (retaining 1 extra leading NIAK frame with $\text{FD}_0 = 0.0$ before the $T_{\text{BOLD}}$ BOLD data rows). The motion-only baseline used this same NIAK motion source.
2. **FD Calculator Correction via NIfTI Volume-Centre Conjugation (`39` → `36` Primary Exclusions; Pre-Label-Join):** During strictly label-free validation against Athena `*_motion.csv` extrema ([`NIAK_VS_ATHENA_MOTION_VALIDATION.md`](NIAK_VS_ATHENA_MOTION_VALIDATION.md)) prior to any diagnosis join, reading world-origin `transf[:3, 3]` showed a translation discrepancy on off-origin NIfTI grids (NeuroIMAGE $\rho = 0.5339$, median ratio `1.4619`; volume center offset `54.5 mm`). Conjugating each scan's transform to its NIfTI volume geometric center (`motion_target_*_session_1_run1.nii.gz`) via $M_{\text{centered}}(t) = T(-\mathbf{c}) M(t) T(+\mathbf{c})$ (commit `3c2b7ac`) raised pooled translation Spearman $\rho$ from `0.8757` to `0.9704` (NeuroIMAGE `0.5339` to `0.8809`) and changed blinded primary exclusions from `39` to `36` (retained `365` to `368`; strict exclusions `299` to `271`). Earlier uncentered FD tables in commit `08f87fe` are marked superseded.
3. **NeuroIMAGE Registration-Failure Scan (~86° NIAK Divergence):** One NeuroIMAGE development scan had a NIAK `minctracc` registration failure ($\max |\theta| = 85.98^\circ$ across `203 / 258` retained frames vs. `7.34°` in Athena `NeuroIMAGE_motion.csv`). Per the pre-specified disposition (no ad-hoc removal), this scan was excluded under the primary FD rule ($N = 368$) and retained in the registered Section 5.4 no-exclusion sensitivity analysis ($N = 404$).
4. **Strict Motion Threshold (`>0.20 mm` OR `>10%` Frames) Restricted to Descriptive Reporting Only:** Under the strict threshold, 3 of 4 LOIO institution folds dropped below 15 subjects per class (`KKI` ADHD = `2`; `NeuroIMAGE` TDC = `11`, ADHD = `8`; `OHSU` TDC = `13`, ADHD = `6`). The strict threshold was therefore kept for descriptive attrition reporting only and was not run as a LOIO classification model.
5. **Wording Correction ("co-primary" → Primary vs. Section 5.4 Sensitivity, Pre-Real-Label Run):** Earlier wording in `DEVIATIONS.md` (`ac517ce`) described the no-exclusion analysis as "co-primary." Prior to any real-label analysis (commit `3063b28`), this was corrected without rewriting Git history: the primary analysis is pooled LOIO ROC-AUC at $N = 368$ (Section 5.3 exclusion), the no-exclusion analysis at $N = 404$ is the registered Section 5.4 sensitivity analysis, and the $N = 404$ cited in Section 7.1 is the preliminary pre-exclusion count from Section 4.2.
6. **Post-Freeze Acquisition of `intermediate_nyu.tar`:** Per [`SEALED_RUN_PLAN.md`](SEALED_RUN_PLAN.md) (`837b33c`), `intermediate_nyu.tar` (NITRC Release 1600) was downloaded after tag `study2-sealed-frozen` (`a75d6b4`) and copied to Git-ignored `NeuroADHD/data/athena/staging/intermediate_nyu.tar` (commit `f21baa1`; size `275,527,680` bytes [`262.76 MB`], self-computed SHA-256 `9640ee0a1623efe91d9eab9ad6a4024d240cdfad3886ecf9b1124a778b141eb4`, `2,665` TAR members / `1,368` files). Zero NYU imaging, time-series, or motion values were inspected or computed before the single sealed execution.

### 5.2. Clarifications (Not Deviations)
1. **190 Populated ROIs Kept and Equivalence Across $N = 404$ and $N = 368$:** Inspection of `templates/ADHD200_parcellate_200.nii.gz` ($49 \times 58 \times 47$) inside `ADHD200_CC200_TCs_filtfix.tar.gz` confirmed `190` populated non-zero ROI labels in `[1, 200]`; `10` cluster IDs (`2, 5, 11, 25, 34, 68, 82, 95, 126, 191`) have `0` voxels and are absent from `3dROIstats` `.1D` output (`192` columns: `File`, `Sub-brick`, plus `190` `Mean_<id>`). All `190 / 190` ROIs had finite, non-zero variance across all `404` development scans (`0` dropped; minimum temporal `std = 0.004031`), yielding `17,955` edges and an identical ROI set for both $N = 404$ and $N = 368$.
2. **Pearson $r$ Clipping Before Fisher $z$:** Off-diagonal correlations were clipped to `[-1 + 1e-7, 1 - 1e-7]` (`FISHER_CLIP_EPS = 1e-7`) prior to $\operatorname{arctanh}(r)$ for numerical stability.
3. **Exact Definition of "Max Displacement":** In `scripts/compute_athena_fd.py` (lines 292–294), computed from volume-centered, first-3-frames-dropped NIAK translations `params[:, 3:6]` (mm) and paired with `mean_fd_mm` as the second feature of `motion_X`:
   ```python
   # Relative translation from first retained frame (Euclidean mm)
   rel_trans = params[:, 3:6] - params[0, 3:6]
   max_disp_mm = float(np.max(np.linalg.norm(rel_trans, axis=1)))
   ```

---

## 6. Limitations

1. **Motion-Associated Attrition and Performance Differences Across Regimes:** Primary FD exclusion removed ADHD scans at a higher rate than TDC scans in the development cohort (`14.8%` vs. `5.0%`, Fisher exact $p = 0.0011$; NeuroIMAGE `24.0%` vs. `0.0%`, $p = 0.0227$), and strict exclusion removed `78.4%` of ADHD scans vs. `59.5%` of TDC scans ($p = 0.0001$). Retaining all 404 scans vs. filtering to 368 scans was accompanied by a difference in fMRI pooled LOIO ROC-AUC (`0.5740`, $p_{\text{perm}} = 0.0050$ at $N = 404$ vs. `0.5241`, $p_{\text{perm}} = 0.2298$ at $N = 368$; Peking `0.6732` vs. `0.5346`). AUC and permutation results differed between the N=404 and N=368 regimes; the cause was not tested.
2. **Non-Imaging Baselines Scored as High or Higher Than fMRI in Every Regime:** Across all four scored regimes, both 2-feature logistic baselines (motion-only and age/sex) scored higher than the locked fMRI pipeline (`primary_368`: fMRI `0.5241` vs. Motion `0.6038` and Age/Sex `0.6163`; `no_exclusion_404`: fMRI `0.5740` vs. Motion `0.6158` and Age/Sex `0.5857`; `primary_sealed`: fMRI `0.5765` vs. Motion `0.6069` and Age/Sex `0.5806`; `sensitivity_no_exclusion`: fMRI `0.5770` vs. Motion `0.6076` and Age/Sex `0.6150`).
3. **Finite-Sample Precision and Statistical Power (Does Not Rule Out a Small Signal):** On sealed NYU, the observed 95% bootstrap CI half-width was approximately $\pm 0.08$ (`[0.496177, 0.656518]` at $N = 196$; `[0.499996, 0.652819]` at $N = 216$). As pre-registered in `PREREGISTRATION.md` Section 7.2, statistical power to detect a true increment of $\Delta\text{AUC} = +0.04$ over age and sex at two-sided $\alpha = 0.05$ is approximately `18%` at $N = 404$ and `26%` at $N = 620$. Accordingly, these intervals and non-significant differences do not rule out a small population effect; they show that under this locked pipeline and sample size, discrimination above chance and incremental value over non-imaging baselines are not established.
4. **Prior Subject Overlap with Study 1 & Development Optimism:** The locked pipeline's hyperparameters were chosen in Study 1 on a cohort that included 126 of the development subjects, so development LOIO estimates may be optimistic for those subjects; only the sealed NYU result is close to independent (6 subjects overlapped).
5. **Cross-Pipeline Motion Substitution, Atlas Divergence, and Scope of Conclusion:** FD was derived from NIAK rigid-body transforms rather than Athena-native AFNI `rp_*.1D` traces (label-free validation pooled Spearman $\rho = 0.9704$ translation, $\rho = 0.9730$ rotation $L_2$, $\rho = 0.9653$ rotation $L_\infty$, with one NeuroIMAGE NIAK registration divergence). Per `PREREGISTRATION.md` Section 9, Study 2 differs from Study 1 in preprocessing pipeline (Athena vs. C-PAC), atlas (`190` populated ROIs vs. `200`), cohort, and sites; accordingly, Study 1's exploratory `0.62` LOIO AUC did not hold up in this pre-registered follow-up on a different preprocessing pipeline, and this conclusion applies to this locked pipeline rather than to functional connectivity in general.
6. **4-Frame Drop FD Sensitivity Check (Not a Protocol Change):** In a label-free sensitivity check dropping 4 rather than 3 leading NIAK frames (`DEVIATIONS.md`, Section 4.6; no classifiers re-run), Section 5.3 primary exclusion status changed for `1 / 404` development scans (`OHSU`: `21` to `22` excluded; `0` changed in KKI, NeuroIMAGE, Peking) and `0 / 216` sealed NYU scans (`20` excluded under both).

---

## 7. Data and Code Availability

- **Primary OSF Preregistration (Complete Binding Protocol):** [https://osf.io/grbt2](https://osf.io/grbt2) — DOI `10.17605/OSF.IO/GRBT2`, registered `2026-11-17` at approximately 14:58 UTC (inferred from the OSF page display), Public, CC0 1.0
- **Open-Ended OSF Registration (Registered First; Metadata & Hashes):** [https://osf.io/db7nv](https://osf.io/db7nv) — DOI `10.17605/OSF.IO/DB7NV` (registered `2026-11-17 14:56 UTC`, Public, CC0 1.0)
- **Locked Protocol File & Registration Commit:** [`docs/study2/PREREGISTRATION.md`](PREREGISTRATION.md), commit `2d7ed3b3be3780c0b8a5b3de7d3c786bb77df19e` (`2d7ed3b`), SHA-256 `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec` (uploaded to OSF `2026-11-17 14:52 UTC`)
- **Chronological Annotated Git Tags (pushed to origin)** *(all descendants of `2d7ed3b`)*:
  1. `study2-dev-analysis-frozen` → `3474758` (`2026-11-18T06:52:18Z`) — frozen development code (`6e812b7`), tests, and feature `.npz` hashes
  2. `study2-dev-results` → `c9a5ca7` (`2026-11-18T07:04:48Z`) — development LOIO results recorded ([`DEV_RESULTS_2026-11-18.md`](DEV_RESULTS_2026-11-18.md))
  3. `study2-sealed-frozen` → `a75d6b4` (`2026-11-18T07:13:05Z`) — frozen sealed-site code (`b96d1ed`) and tests prior to NYU access
  4. `study2-sealed-results` → `65baffc` (`2026-11-18T07:23:56Z`) — one-shot sealed NYU results recorded ([`SEALED_RESULTS_2026-11-18.md`](SEALED_RESULTS_2026-11-18.md))
- **Git-Ignored Result Artifact SHA-256 Checksums:**
  - Development JSON & log (`study2_dev_results.json`, `study2_dev_run.log`): `6d072448bf6573cd7f6e0bd65c0b6a141bc2cf60f2ea0e4a69bcc504918dc0d5`
  - Sealed JSON & log (`study2_sealed_results.json`, `study2_sealed_run.log`): `117e0395e73f9c9e0388d9a37d6e7de1405c052cfdc2c35769891cbc7d548dc2`
- **Source Data Access:** ADHD-200 preprocessed derivatives (Athena CC200 and NIAK intermediate motion archives) are distributed by the ADHD-200 Consortium and the Neuro Bureau via NITRC Release 1600 and are excluded from version control (`.gitignore`).

### Verifiability of the Timeline

Only the public OSF registrations (`https://osf.io/db7nv` and `https://osf.io/grbt2`, together with the uploaded `PREREGISTRATION.md` file and its SHA-256 hash) are externally timestamped by a third party. The four milestone tags (`study2-dev-analysis-frozen`, `study2-dev-results`, `study2-sealed-frozen`, `study2-sealed-results`) and branch commits were pushed to `origin` after the sealed NYU run had completed; accordingly, the ordering and timestamps of the intermediate freezes and runs are documented in local Git history rather than independently verified by an external timestamping authority.
