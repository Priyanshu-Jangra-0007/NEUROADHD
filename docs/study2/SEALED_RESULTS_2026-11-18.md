# Study 2 Sealed External Validation Results — NYU (2026-11-18)

**Date:** 2026-11-18  
**Branch:** `feat/study2-athena-acquisition`  
**Preregistration Reference:** OSF Preregistration ([https://osf.io/grbt2](https://osf.io/grbt2), DOI: `10.17605/OSF.IO/GRBT2`)  
**Locked Protocol SHA-256 (unmodified):** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`  
**Frozen Sealed Code Commit:** `b96d1edead9b59e54b755fc8abf18f8a16a56ea8` (`scripts/run_study2_sealed_site.py`, SHA-256: `aa0bdc9f00a7d97e8afa41621eeafbc78f97fc9e4507333cbd4f89acb4fbddf3`)  
**Sealed Freeze Commit / Tag:** `a75d6b47e33dc994ee552a1a2fff1bac3e4a3b11` (`study2-sealed-frozen`)  
**Pre-Run NYU Archive Staging Commit:** `f21baa1` (`intermediate_nyu.tar`, 275,527,680 bytes, SHA-256: `9640ee0a1623efe91d9eab9ad6a4024d240cdfad3886ecf9b1124a778b141eb4`)

> Results reported as produced by the frozen script; no analysis choices were changed after seeing them.
>
> Single sealed scoring; nothing re-tuned or re-run.

---

## 1. Execution Artifacts & Separate Checksums (Git-Ignored under `NeuroADHD/data/athena/results/`)

- **Command Executed (Once, Locked Defaults):** `PYTHONPATH=. ./NeuroADHD/.venv/bin/python scripts/run_study2_sealed_site.py --execute-real-sealed`
- **Exit Code:** `0` (zero errors, zero retries)
- **Results JSON (`NeuroADHD/data/athena/results/study2_sealed_results.json`, 1,872 bytes, inode `12191721`) SHA-256:**  
  `117e0395e73f9c9e0388d9a37d6e7de1405c052cfdc2c35769891cbc7d548dc2`
- **Run Log (`NeuroADHD/data/athena/results/study2_sealed_run.log`, 1,872 bytes, inode `12191704`) SHA-256:**  
  `117e0395e73f9c9e0388d9a37d6e7de1405c052cfdc2c35769891cbc7d548dc2`
- **Seeds & $B$ Values:** RF `random_state = 42`, logistic baselines `random_state = 42`, stratified bootstrap `seed = 42`, `B = 2,000`, `k = 100`.

---

## 2. Sealed NYU Cohort Counts (Aggregate Only)

| Metric | Total | TDC | ADHD | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **NYU Complete `rest_1` Scans** | **216** | **98** | **118** | 100% match to CC200 archive member names |
| **Excluded by Section 5.3 Primary FD Rule** | **20** | **7** (7.1%) | **13** (11.0%) | Two-sided Fisher exact $p = 0.3568$ |
| **Retained (Primary Sealed Regime)** | **196** | **91** | **105** | Scored in `primary_sealed` |
| **Section 5.1 Zero-Filled ROIs** | **0** | — | — | `0` scans / `0` ROI instances zero-filled across all 216 scans |

---

## 3. Primary Sealed Analysis — `primary_sealed` (Governs Section 8.2 Outcome Zone)

Refitted once on pooled `primary_368` ($N = 368$, 17,955 edges, $k = 100$) and scored once on primary-FD-filtered NYU ($N = 196$: 91 TDC / 105 ADHD).

| Model / Endpoint | NYU ROC-AUC | 95% Stratified Bootstrap CI ($B = 2,000$, seed `42`) | Section 8.2 Outcome Zone |
| :--- | :---: | :---: | :---: |
| **Locked fMRI Pipeline (Primary — Governing)** | **0.5765** (`0.576452`) | **`[0.4962, 0.6565]`** (`[0.496177, 0.656518]`) | **`inconclusive`** |
| **Motion-Only Baseline *(descriptive only)*** | **0.6069** (`0.606907`) | `[0.5276, 0.6805]` (`[0.527556, 0.680484]`) | — |
| **Age & Sex Baseline *(descriptive only)*** | **0.5806** (`0.580638`) | `[0.5000, 0.6593]` (`[0.500044, 0.659267]`) | — |

- **Mechanical Section 8.2 Classification (Governing):** Point estimate `0.576452 > 0.50` and 95% bootstrap CI lower bound `0.496177 <= 0.50` $\implies$ **`inconclusive`** (*Inconclusive Zone*).

---

## 4. Sensitivity Sealed Analysis — `sensitivity_no_exclusion` (Registered Section 5.4)

Refitted once on pooled `no_exclusion_404` ($N = 404$, 17,955 edges, $k = 100$) and scored once on all complete NYU `rest_1` scans ($N = 216$: 98 TDC / 118 ADHD).

| Model / Endpoint | NYU ROC-AUC | 95% Stratified Bootstrap CI ($B = 2,000$, seed `42`) | Section 8.2 Outcome Zone |
| :--- | :---: | :---: | :---: |
| **Locked fMRI Pipeline (Sensitivity)** | **0.5770** (`0.576963`) | **`[0.5000, 0.6528]`** (`[0.499996, 0.652819]`) | **`inconclusive`** |
| **Motion-Only Baseline *(descriptive only)*** | **0.6076** (`0.607575`) | `[0.5335, 0.6825]` (`[0.533531, 0.682467]`) | — |
| **Age & Sex Baseline *(descriptive only)*** | **0.6150** (`0.615012`) | `[0.5367, 0.6928]` (`[0.536651, 0.692775]`) | — |

- **Mechanical Section 8.2 Classification (Sensitivity):** Point estimate `0.576963 > 0.50` and 95% bootstrap CI lower bound `0.499996 <= 0.50` $\implies$ **`inconclusive`** (*Inconclusive Zone*).
