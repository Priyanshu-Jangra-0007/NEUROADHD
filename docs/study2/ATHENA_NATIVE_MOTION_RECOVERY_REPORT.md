# Athena-Native Motion Archive Inspection & Development FD Report (Study 2 — Phase 2G)

**Document Type:** Acquisition, Inspection & Blinded FD Summary Report  
**Branch:** `feat/study2-athena-acquisition`  
**Registration Reference:** OSF Preregistration ([https://osf.io/grbt2](https://osf.io/grbt2), DOI: `10.17605/OSF.IO/GRBT2`)  
**Locked Protocol Hash (SHA-256, verified against OSF & commit `2d7ed3b`):**  
`e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`  
**Pre-Inspection Deviation Rule Commit:** `ac517ce` ([`docs/study2/DEVIATIONS.md`](DEVIATIONS.md), committed **before** opening any `_preproc_xfms.tar` archive)  
**Date:** 2026-11-18  
**Guardrails:** Zero individual subject IDs printed or logged; zero diagnosis labels joined; zero NYU archives downloaded, extracted, or read; zero modeling executed.

---

## 1. Executive Decision Summary

- **`_preproc_xfms.tar` Inspection Result:** All six development-site `<Site>_preproc_xfms.tar` archives (54.83 MB total) were staged, verified intact without extraction, and inspected. They contain **only** FSL FNIRT structural-to-template warp fields (`<Site>//<ID>/<ID>_session_1_template.nii.gz`) and **0 / 404 `rp_*.1D` files**.
- **Pre-Specified Rule Activated (`DEVIATIONS.md` Branch b):** Because fewer than 404 Athena-native AFNI `rp_*.1D` files are present, AFNI extraction stopped (0 files extracted from `_preproc_xfms.tar`) and **Protocol Deviation Branch (b)** was activated:
  - Power et al. (2012) FD (50 mm radius, $\text{FD}_0 = 0$) computed from the staged Neuro Bureau NIAK `motion_parameters_*.mat` $4 \times 4 \times T$ rigid-body matrices with the **first 3 frames removed** (0 frame-count mismatches across all 404 development CC200 `rest_1` scans).
  - Labelled an explicit **protocol deviation**, with the pre-registered **no-exclusion analysis** elevated to a **co-primary sensitivity analysis**.
  - The motion-only baseline will use the same NIAK-derived motion source.

---

## 2. Staged `<Site>_preproc_xfms.tar` Archives & Member Audit

All six archives were moved cleanly from `/Users/apple/Downloads/NeuroAdhd NITRC` (now empty) into `NeuroADHD/data/athena/staging/` with zero overwrites.  
*(Note: All SHA-256 checksums below are **self-computed locally** after download for provenance tracking; NITRC Release 1600 does not publish SHA-256 checksums.)*

| # | Archive Filename | Size (Bytes) | Human Size | Self-Computed SHA-256 | TAR Integrity | Total Entries | `rp_*.1D` Count | Masked Member Pattern |
| :-: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `KKI_preproc_xfms.tar` | 11,223,040 | 10.70 MB | `d50c8c75245661547502553fd80d0dddb89b1540cc24aefaba2649172acf16fb` | **PASS** | 83 | **0** | `KKI//<ID>/<ID>_session_1_template.nii.gz` (83) |
| 2 | `NeuroIMAGE_preproc_xfms.tar` | 6,522,880 | 6.22 MB | `5db89c94e1fb0e78c65a91e297aba17bd1b4b77d9eefbff0a1f1ddd25d42e652` | **PASS** | 48 | **0** | `NeuroIMAGE//<ID>/<ID>_session_1_template.nii.gz` (48) |
| 3 | `OHSU_preproc_xfms.tar` | 10,721,280 | 10.22 MB | `3ef125096dba115baff2a3b1207df2dc720ab79a9417740e7eaac7080e331d94` | **PASS** | 79 | **0** | `OHSU//<ID>/<ID>_session_1_template.nii.gz` (79) |
| 4 | `Peking_1_preproc_xfms.tar` | 11,550,720 | 11.02 MB | `e07383b1015e8a83b90e97c3475c1b96d7bd867b713244dda5132f6a2c82709e` | **PASS** | 85 | **0** | `Peking_1//<ID>/<ID>_session_1_template.nii.gz` (85) |
| 5 | `Peking_2_preproc_xfms.tar` | 9,103,360 | 8.68 MB | `08436a79a4d806791317ac00cc96436ea62169e55c3643ae9adf8ef5cdd6bf92` | **PASS** | 67 | **0** | `Peking_2//<ID>/<ID>_session_1_template.nii.gz` (67) |
| 6 | `Peking_3_preproc_xfms.tar` | 5,713,920 | 5.45 MB | `5129765c70ff1b3d4227c28fecddd36047d01c0c5b606d160f1b02dee9b0ea3d` | **PASS** | 42 | **0** | `Peking_3//<ID>/<ID>_session_1_template.nii.gz` (42) |
| **Total** | — | **54,835,200** | **52.30 MB** | — | **6/6 PASS** | **404** | **0 / 404** | **Warp fields only; zero files extracted** |

---

## 3. Development-Site Framewise Displacement Summary (Branch b — Protocol Deviation, No Diagnosis Labels)

Computed via [`scripts/compute_athena_fd.py`](../../scripts/compute_athena_fd.py) (output written to Git-ignored `NeuroADHD/data/athena/motion/dev_niak_fd_metrics.csv`):
- **Motion Source:** Staged NIAK `intermediate_*.tar` `motion_parameters_*_session_1_run1.mat` ($4 \times 4 \times T_{\text{raw}}$ rigid-body matrices, $\det(R) = 1.0$, orthogonality error $< 10^{-4}$), with the **first 3 frames dropped** (`transf[:, :, 3:]`).
- **CC200 Frame-Count Alignment (Development Sites Only):** **404 / 404 matched** with **0 frame-count mismatches** (NYU, WashU, and Pittsburgh skipped completely).
- **Power et al. (2012) Formula:** Sphere radius $r = 50\text{ mm}$, $\text{FD}_0 = 0.0$ per `PREREGISTRATION.md` Section 5.3.
- **Diagnosis Labels Joined:** **NONE.**

### 3.1. Primary Exclusion Threshold ($\text{Mean FD} > 0.50\text{ mm}$ OR $>20\%$ frames with $\text{FD} > 0.50\text{ mm}$)

| Site | Scans ($N$) | $\text{Mean FD} > 0.50\text{ mm}$ | $>20\%$ Frames $> 0.50\text{ mm}$ | Primary Excluded (OR) | Primary Retained |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KKI** | 83 | 2 | 4 | **4** | 79 |
| **NeuroIMAGE** | 48 | 6 | 5 | **6** | 42 |
| **OHSU** | 79 | 19 | 23 | **23** | 56 |
| **Peking_1** | 85 | 1 | 2 | **2** | 83 |
| **Peking_2** | 67 | 3 | 3 | **3** | 64 |
| **Peking_3** | 42 | 0 | 1 | **1** | 41 |
| *(Peking Combined)* | *(194)* | *(4)* | *(6)* | *(6)* | *(188)* |
| **TOTAL (Dev)** | **404** | **31** | **38** | **39** | **365** |

### 3.2. Stricter Sensitivity Threshold ($\text{Mean FD} > 0.20\text{ mm}$ OR $>10\%$ frames with $\text{FD} > 0.20\text{ mm}$)

| Site | Scans ($N$) | $\text{Mean FD} > 0.20\text{ mm}$ | $>10\%$ Frames $> 0.20\text{ mm}$ | Strict Excluded (OR) | Strict Retained |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KKI** | 83 | 30 | 62 | **62** | 21 |
| **NeuroIMAGE** | 48 | 13 | 36 | **36** | 12 |
| **OHSU** | 79 | 41 | 59 | **59** | 20 |
| **Peking_1** | 85 | 14 | 63 | **63** | 22 |
| **Peking_2** | 67 | 11 | 40 | **40** | 27 |
| **Peking_3** | 42 | 18 | 39 | **39** | 3 |
| *(Peking Combined)* | *(194)* | *(43)* | *(142)* | *(142)* | *(52)* |
| **TOTAL (Dev)** | **404** | **127** | **299** | **299** | **105** |

### 3.3. Co-Primary No-Exclusion Sensitivity & Denominator Check
- **No-Exclusion Co-Primary Sensitivity (`DEVIATIONS.md` Branch b):** **404 / 404** development scans retained (0 excluded).
- **Denominator Check ($T$ including $\text{FD}_0 = 0$ vs. $T - 1$ excluding $\text{FD}_0$):** Primary exclusions = 39 (denominator $T$, matching `PREREGISTRATION.md` Section 5.3) vs. 40 (denominator $T - 1$); strict exclusions = 299 under both denominators.

---

## 4. Boundary & Quarantine Confirmation
- **NYU Quarantine:** **PASS.** Zero NYU archives downloaded, inspected, extracted, or read (`load_dev_cc200_frame_counts` explicitly skips `NYU`, `WashU`, and `Pittsburgh`).
- **Preregistration Immutability:** `docs/study2/PREREGISTRATION.md` untouched (`e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`).
- **Label Blinding & Modeling:** Zero diagnosis labels joined; zero models trained.
- **Git Ignore Protection:** `NeuroADHD/data/athena/staging/` and `NeuroADHD/data/athena/motion/` confirmed ignored by `.gitignore` rule `**/athena/`.
