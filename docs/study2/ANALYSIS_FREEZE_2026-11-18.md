# Study 2 Development Analysis Freeze (2026-11-18)

**Date:** 2026-11-18  
**Branch:** `feat/study2-athena-acquisition`  
**Local Annotated Tag:** `study2-dev-analysis-frozen` *(local only; not pushed)*  
**Status:** **FROZEN PRIOR TO ANY REAL-LABEL ANALYSIS**  
**Preregistration Reference:** OSF Preregistration ([https://osf.io/grbt2](https://osf.io/grbt2), DOI: `10.17605/OSF.IO/GRBT2`)  
**Locked Protocol File (Unmodified):** [`docs/study2/PREREGISTRATION.md`](PREREGISTRATION.md)  
**Locked Protocol SHA-256:** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`

> No methodological change will be made after the first real-label run; crash fixes only, logged in DEVIATIONS.md.

---

## 1. Frozen Analysis Script & Synthetic Test Suite

- **Frozen Code Commit Hash:** `6e812b7ddbd34321ad767c645fe5db3c7235a7ef` (`6e812b7`)

| File | Commit Hash | Self-Computed SHA-256 |
| :--- | :---: | :--- |
| `scripts/run_study2_dev_analysis.py` | `6e812b7ddbd34321ad767c645fe5db3c7235a7ef` | `fa4b41d2966e8219e93ac3a6151b1c141d10ef34af167771607edb88820d50b1` |
| `tests/test_run_study2_dev_analysis.py` | `6e812b7ddbd34321ad767c645fe5db3c7235a7ef` | `89d4745453cf079d4b6e70a9a54e3412fa9f1531c247a185995707acac9693df` |

- **Default Output Path (Git-Ignored under `.gitignore:33:**/athena/`):** `NeuroADHD/data/athena/results/study2_dev_results.json`

---

## 2. Frozen Development Feature Files (Git-Ignored under `NeuroADHD/data/athena/features/`)

| File | Shape / Scope | Size (Bytes) | Self-Computed SHA-256 |
| :--- | :--- | :---: | :--- |
| `NeuroADHD/data/athena/features/dev_cc200_fisher_z_all404.npz` | `(404, 17955)` — Section 5.4 no-exclusion sensitivity | 55,579,205 | `1da75f4368654d52d4ad0c9a002aa287949f738c5030b24ace868e0aff0e47e4` |
| `NeuroADHD/data/athena/features/dev_cc200_fisher_z_primary368.npz` | `(368, 17955)` — Section 5.3 primary analysis | 50,626,028 | `a5727a7a9d7318d4d60013f4ce89ff7aaf9ae2ab194b15b3f9b20dc1cbe90baa` |
| `NeuroADHD/data/athena/features/dev_valid_roi_set.json` | 190 / 190 ROIs kept (0 dropped; 17,955 edges) | 4,847 | `ba5ce47259587f762d575e792cd0cdaf7c754c7b1b508efc817c57348829b133` |

---

## 3. Frozen Random Seeds, Resample Counts ($B$), and Hyperparameters

| Component | Parameter / Seed | Locked Value | Source Constant / Line |
| :--- | :--- | :---: | :--- |
| **Random Forest Classifier** | `n_estimators`, `max_depth`, `class_weight`, `random_state` | `50`, `3`, `'balanced'`, **`42`** | `LOCKED_RANDOM_STATE = 42` |
| **ANOVA Feature Selection** | `score_func`, `k` | `f_classif`, **`100`** | `DEFAULT_K_BEST = 100` |
| **Motion-Only & Age/Sex Baselines** | `LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000)` | **`42`** | `LOCKED_RANDOM_STATE = 42` |
| **Stratified Bootstrap (H1 Pooled & Per-Institution CIs)** | `n_boot` ($B$), `seed` | **$B = 2,000$**, **`42`** | `DEFAULT_BOOTSTRAP_B = 2000`, `seed = 42` |
| **Paired Stratified Bootstrap (H2 vs. Motion, H3 vs. Age/Sex)** | `n_boot` ($B$), `seed` | **$B = 2,000$**, **`42`** | `DEFAULT_BOOTSTRAP_B = 2000`, `seed = 42` |
| **Within-Institution Permutation Test (H1 $p_{\text{perm}}$)** | `n_perm` ($B$), `seed`, RF `random_state` | **$B = 1,000$**, **`42`**, RF **`42`** | `DEFAULT_PERM_B = 1000`, `seed = 42` |

---

## 4. Registered Endpoint & Regime Coverage Confirmed

- **H1:** Pooled LOIO ROC-AUC + within-institution permutation $p_{\text{perm}}$ ($B = 1,000$) + stratified bootstrap 95% CI ($B = 2,000$).
- **H2:** Paired stratified bootstrap $\Delta\text{AUC}_{\text{motion}} = \text{AUC}_{\text{fMRI}} - \text{AUC}_{\text{motion}}$ ($B = 2,000$, outperformance iff $\text{CI}_{\text{lo}} > 0$).
- **H3:** Paired stratified bootstrap $\Delta\text{AUC}_{\text{age/sex}} = \text{AUC}_{\text{fMRI}} - \text{AUC}_{\text{Age/Sex}}$ ($B = 2,000$).
- **Per-Institution AUCs:** Individual stratified bootstrap 95% CIs across all four LOIO institutions (`KKI`, `NeuroIMAGE`, `OHSU`, `Peking`).
- **Regimes:** Primary analysis (`N = 368`, Section 5.3) and registered sensitivity analysis (`N = 404`, Section 5.4).
- **NYU Quarantine:** Sealed external validation institution (`N = 216`: 98 TDC, 118 ADHD from CC200 member names only) remains untouched and unopened.
