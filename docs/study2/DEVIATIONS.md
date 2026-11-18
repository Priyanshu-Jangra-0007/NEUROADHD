# Study 2 Protocol Deviations & Pre-Inspection Fallback Rules

**Date:** 2026-11-18  
**Branch:** `feat/study2-athena-acquisition`  
**Preregistration Reference:** OSF Preregistration ([https://osf.io/grbt2](https://osf.io/grbt2), DOI: `10.17605/OSF.IO/GRBT2`)  
**Locked Protocol File (Unmodified):** [`docs/study2/PREREGISTRATION.md`](PREREGISTRATION.md)  
**Locked Protocol SHA-256:** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`

> [!IMPORTANT]
> This file is authored and committed **prior to opening or inspecting** the six development-site Athena `<Site>_preproc_xfms.tar` archives staged from NITRC Release 1600. `docs/study2/PREREGISTRATION.md` is not modified.

---

## 1. Motion Parameter Acquisition & Framewise Displacement (FD) Fallback Rule

To preserve prospective analytical discipline while resolving the availability of Athena-native framewise realignment traces (`rp_*.1D`), the following deterministic decision rule is locked prior to inspecting `<Site>_preproc_xfms.tar`:

### Branch (a) — Pre-Registered Primary Path (Athena-Native AFNI `rp_*.1D`)
- **Condition:** Authentic Athena-native AFNI `rp_*.1D` six-parameter framewise motion traces are present and validated for **all 404 development `rest_1` scans** across KKI ($N = 83$), NeuroIMAGE ($N = 48$), OHSU ($N = 79$), and Peking 1–3 ($N = 194$).
- **Action:** Extract only those `rp_*.1D` files into `NeuroADHD/data/athena/motion/`, validate 6 numeric columns (`roll`, `pitch`, `yaw`, `dS`, `dL`, `dP`), units (degrees and millimeters), and frame-count match against the Athena CC200 `rest_1` time series, and compute Power et al. (2012) FD (50 mm sphere radius) exactly as pre-registered.

### Branch (b) — Protocol Deviation Fallback Path (NIAK-Derived FD)
- **Condition:** Fewer than 404 Athena-native AFNI `rp_*.1D` traces are present in the six development `<Site>_preproc_xfms.tar` archives.
- **Action:**
  1. Stop AFNI `rp_*.1D` extraction and report exact site counts.
  2. Compute Power et al. (2012) FD (50 mm sphere radius) from the staged Neuro Bureau NIAK `motion_parameters_*.mat` $4 \times 4 \times T$ rigid-body transformation matrices with the **first 3 frames removed** (aligning length $T_{\text{NIAK}} - 3 = T_{\text{CC200}}$ across all 404 development scans).
  3. Explicitly label this substitution as a **protocol deviation** in all downstream Study 2 reports and manuscripts.
  4. Elevate the pre-registered **no-exclusion analysis** (retaining all scans without FD-based exclusion) to a **co-primary sensitivity analysis** alongside the FD-filtered analysis.

---

## 2. Motion-Only Baseline & Label Blinding Rules

1. **Motion-Only Baseline Parity:** The pre-registered motion-only baseline model uses **whichever motion source is selected by the rule above** — Branch (a) AFNI `rp_*.1D` if all 404 exist, or Branch (b) NIAK-derived parameters (first 3 frames removed) otherwise.
2. **Strict Blinding to Diagnosis Labels:** **No FD computation, threshold check, alignment check, or fallback decision uses or joins diagnosis labels (`DX`).** All motion extraction and FD thresholding are executed strictly blind to diagnostic class.
3. **NYU Quarantine:** This rule applies exclusively to development-site scans (KKI, NeuroIMAGE, OHSU, Peking 1–3). No NYU archive is opened, extracted, or analyzed during development.
