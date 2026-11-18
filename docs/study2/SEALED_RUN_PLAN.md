# Study 2 Sealed External Validation Plan — NYU (Written Prior to Any NYU Data Read)

**Date:** 2026-11-18  
**Branch:** `feat/study2-athena-acquisition`  
**Status:** **LOCKED PLAN — ZERO NYU IMAGING, MOTION, OR FEATURES READ**  
**Preregistration Reference:** OSF Preregistration ([https://osf.io/grbt2](https://osf.io/grbt2), DOI: `10.17605/OSF.IO/GRBT2`)  
**Locked Protocol SHA-256 (unmodified):** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`  
**Development Freeze Reference:** `study2-dev-analysis-frozen` (`3474758`) & `study2-dev-results` (`c9a5ca7`)

> [!IMPORTANT]
> This plan is written and committed **before** any NYU imaging, time-series, or NIAK motion transform file is downloaded or read. Nothing in the locked pipeline, seeds, features, or ROI set has been modified in response to development results.

---

## 1. Exact One-Shot Sealed Regimes (`PREREGISTRATION.md`, Sections 4.3, 5.3, 5.4, 8.2)

### 1.1. Primary Sealed Analysis (Governs the Section 8.2 Outcome Zone)
1. Refit the locked 4-step pipeline + `RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42)` (imported directly from `scripts/run_study2_dev_analysis.py` without modification) **exactly once** on the pooled `primary_368` development set ($N = 368$, 17,955 Fisher-$z$ edges).
2. Apply the registered Section 5.3 primary FD exclusion rule ($\text{Mean FD} > 0.50\text{ mm}$ OR $> 20\%$ frames with $\text{FD} > 0.50\text{ mm}$) to NYU.
3. Score the frozen refitted model **exactly once** on the retained NYU primary subset.
4. This primary sealed analysis **strictly governs** the Section 8.2 outcome zone classification.

### 1.2. Registered No-Exclusion Sensitivity Sealed Analysis
1. Refit the identical locked pipeline **exactly once** on the pooled `no_exclusion_404` development set ($N = 404$, 17,955 Fisher-$z$ edges).
2. Score the frozen refitted model **exactly once** on **all** NYU `rest_1` scans with complete time series ($N = 216$ expected from CC200 member names).
3. Report alongside the primary sealed analysis as the registered Section 5.4 sensitivity analysis.

---

## 2. Frozen ROI Set & Sealed-Site Zero-Edge Rule (`PREREGISTRATION.md`, Section 5.1)

- Use **strictly** the frozen 190-ROI development set recorded in `NeuroADHD/data/athena/features/dev_valid_roi_set.json` (identical for both $N = 368$ and $N = 404$; $P = 17,955$ upper-triangle edges).
- If any ROI in the 190-ROI development set is missing, non-finite, or constant ($\text{std} = 0$) in an NYU subject:
  - Compute pairwise Pearson correlations and Fisher $z$ values (clipped to `[-1 + 1e-7, 1 - 1e-7]`) across all non-constant ROIs for that subject, and set **every Fisher-$z$ edge incident to the missing/constant ROI to `0.0`** (representing prior mean $r = 0, z = 0$).
  - **Never** drop the ROI from the feature space, **never** re-select the top-100 ANOVA edges, and **never** refit model weights.
- Log aggregate counts only (number of NYU scans and ROI instances triggering zero-edge imputation; zero subject IDs).

---

## 3. NYU Motion & Data Access Boundaries

- **NYU FD Computation:** Compute NYU Framewise Displacement using the **frozen** `scripts/compute_athena_fd.py` logic (`read_nifti_volume_center_world`, `center_shift_transf`, `parse_octave_transf_text`, `extract_6dof_params`, `compute_power_fd`, dropping the first 3 frames, $r = 50\text{ mm}$, $\text{FD}_0 = 0.0$, `max_disp_mm` relative to frame 0) — **zero code modifications** to `scripts/compute_athena_fd.py`.
- **Source Separation & Access Guard:**
  - NYU motion transform archive (`intermediate_nyu.tar`, NITRC Release 1600 `/frs/downloadlink.php/3808`) may be read strictly to compute label-blind NYU FD.
  - NYU imaging/time-series members (`NYU/<ID>/sfnwmrda<ID>_session_1_rest_1_cc200_TCs.1D` in `ADHD200_CC200_TCs_filtfix.tar.gz`) may be read **exclusively by `scripts/run_study2_sealed_site.py`**, and only when **both** `--execute-real-sealed` is passed **and** local annotated tag `study2-sealed-frozen` is an ancestor of `HEAD`.
  - The sealed runner writes results strictly to a Git-ignored path (`NeuroADHD/data/athena/results/study2_sealed_results.json`) and **refuses to overwrite** an existing results file.

---

## 4. Metrics, Outcome Zones & Reporting Requirements

### 4.1. Primary & Sensitivity fMRI Endpoints
- **NYU ROC-AUC** + **stratified bootstrap 95% CI** ($B = 2,000$ resamples, `seed = 42`, stratified by binary diagnosis `y`).
- **Section 8.2 Outcome Zone** (governed by the **primary** sealed analysis):

| Outcome Zone Code | Statistical Condition (`PREREGISTRATION.md`, Section 8.2) |
| :--- | :--- |
| **`confirmed`** | 95% bootstrap CI lower bound $> 0.50$ (**Confirmed Generalization**) |
| **`inconclusive`** | Point estimate $> 0.50$, and 95% bootstrap CI lower bound $\le 0.50$ (**Inconclusive Zone**) |
| **`no_evidence`** | Point estimate $\le 0.50$ (**No Evidence of Transfer**) |

### 4.2. Descriptive Baselines (Reported as Descriptive Only)
- **Motion-Only Baseline:** `build_linear_baseline_pipeline()` (`SimpleImputer -> StandardScaler -> LogisticRegression(C=1.0, class_weight='balanced', random_state=42)`) fit once on the matching pooled development set (`[mean_fd_mm, max_disp_mm]`) and scored once on NYU, reporting ROC-AUC and stratified bootstrap 95% CI ($B = 2,000$, `seed = 42`).
- **Age & Sex Baseline:** Same pipeline fit once on the matching pooled development set (`[Age, Gender]`) and scored once on NYU, reporting ROC-AUC and stratified bootstrap 95% CI ($B = 2,000$, `seed = 42`).

### 4.3. Required Aggregate Cohort Counts (Zero Subject IDs)
- Total NYU `rest_1` scans with complete time series and valid diagnosis.
- NYU scans excluded under the Section 5.3 primary FD rule, broken down by diagnosis (TDC excluded, ADHD excluded, plus two-sided Fisher exact $p$).
- NYU scans retained under the primary rule, broken down by diagnosis (TDC retained, ADHD retained).
- Count of NYU scans and ROI instances invoking Section 5.1 zero-edge imputation.

---

## 5. One-Shot Finality Rule

**Nothing is re-selected, re-tuned, refitted, or re-run after the single sealed scoring.** Once `scripts/run_study2_sealed_site.py --execute-real-sealed` completes and writes `NeuroADHD/data/athena/results/study2_sealed_results.json`, the sealed external validation is permanently closed and reported verbatim.
