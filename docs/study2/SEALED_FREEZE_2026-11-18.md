# Study 2 Sealed External Validation Freeze (2026-11-18)

**Date:** 2026-11-18  
**Branch:** `feat/study2-athena-acquisition`  
**Local Annotated Tag:** `study2-sealed-frozen` *(local only; not pushed)*  
**Status:** **FROZEN PRIOR TO READING ANY NYU IMAGING, MOTION, OR FEATURES**  
**Preregistration Reference:** OSF Preregistration ([https://osf.io/grbt2](https://osf.io/grbt2), DOI: `10.17605/OSF.IO/GRBT2`)  
**Locked Protocol File (Unmodified):** [`docs/study2/PREREGISTRATION.md`](PREREGISTRATION.md)  
**Locked Protocol SHA-256:** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`  
**Sealed Plan Reference:** [`docs/study2/SEALED_RUN_PLAN.md`](SEALED_RUN_PLAN.md) (`837b33c`)

> No methodological change will be made after the sealed run; crash fixes only, logged in DEVIATIONS.md.

---

## 1. Frozen Sealed-Site Script & Synthetic Test Suite

- **Frozen Sealed Code Commit Hash:** `b96d1edead9b59e54b755fc8abf18f8a16a56ea8` (`b96d1ed`)

| File | Commit Hash | Self-Computed SHA-256 |
| :--- | :---: | :--- |
| `scripts/run_study2_sealed_site.py` | `b96d1edead9b59e54b755fc8abf18f8a16a56ea8` | `aa0bdc9f00a7d97e8afa41621eeafbc78f97fc9e4507333cbd4f89acb4fbddf3` |
| `tests/test_run_study2_sealed_site.py` | `b96d1edead9b59e54b755fc8abf18f8a16a56ea8` | `56c1e9e7038680e018572dc53eb82d6aee25b9271985e7275d2915740e201b06` |

- **Unchanged Imported Frozen Dependencies:**
  - `scripts/run_study2_dev_analysis.py` (`6e812b7`): `fa4b41d2966e8219e93ac3a6151b1c141d10ef34af167771607edb88820d50b1`
  - `scripts/compute_athena_fd.py` (`3c2b7ac`): `befb5c8f5f4022806e9047be4d2a4fdea9dff5a12be68849926c912edd5550a5`
- **Default Output Path (Git-Ignored under `.gitignore:33:**/athena/`):** `NeuroADHD/data/athena/results/study2_sealed_results.json` (overwrite refused if present).

---

## 2. Frozen Random Seeds, Resample Counts ($B$), and Hyperparameters

| Component | Parameter / Seed | Locked Value | Source Constant |
| :--- | :--- | :---: | :--- |
| **Random Forest Classifier** | `n_estimators`, `max_depth`, `class_weight`, `random_state` | `50`, `3`, `'balanced'`, **`42`** | `LOCKED_RANDOM_STATE = 42` |
| **ANOVA Feature Selection** | `score_func`, `k` | `f_classif`, **`100`** | `DEFAULT_K_BEST = 100` |
| **Descriptive Baselines (Motion-Only, Age & Sex)** | `LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000)` | **`42`** | `LOCKED_RANDOM_STATE = 42` |
| **Stratified Bootstrap 95% CIs (NYU fMRI & Baselines)** | `n_boot` ($B$), `seed` | **$B = 2,000$**, **`42`** | `DEFAULT_BOOTSTRAP_B = 2000`, `seed = 42` |
| **FD Transformation Alignment** | Volume-centered NIAK `transf`, `DUMMY_FRAMES_DROPPED`, `HEAD_RADIUS_MM` | `3` frames, `50.0 mm`, `FD_0 = 0.0` | Frozen `scripts/compute_athena_fd.py` |

---

## 3. Hard Execution Guards & One-Shot Lock

1. **Flag Guard:** Refuses to read NYU data without `--execute-real-sealed`.
2. **Tag Guard:** Refuses to read NYU data unless local annotated tag `study2-sealed-frozen` is an ancestor of `HEAD`.
3. **Overwrite Guard:** Refuses to run if `NeuroADHD/data/athena/results/study2_sealed_results.json` already exists.
4. **NYU Quarantine Status at Freeze:** Zero NYU imaging, time series, or NIAK motion archives have been downloaded or read.
