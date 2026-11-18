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

---

## 3. Dated Deviation & Implementation Log — 2026-11-18

### 3.1. Activation of Fallback Branch (b)
- All six development `<Site>_preproc_xfms.tar` archives contained FSL FNIRT structural-to-template warp fields only (`0 / 404` AFNI `rp_*.1D` traces; commit `08f87fe`).
- Per Section 1 above, **Branch (b)** was activated: NIAK-derived Power et al. (2012) FD with the first 3 frames removed, labelled a protocol deviation, with no-exclusion elevated to a co-primary sensitivity analysis.

### 3.2. FD Calculator Correction: NIfTI Volume-Centre Conjugation (Pre-Label-Join)
- **Discovery (Strictly Label-Free):** During label-free validation of NIAK-derived motion against the official Athena `*_motion.csv` extrema ([`NIAK_VS_ATHENA_MOTION_VALIDATION.md`](NIAK_VS_ATHENA_MOTION_VALIDATION.md)), executed **before any diagnosis labels were joined**, naive extraction of world-origin translation `transf[:3, 3]` exhibited a severe translation discrepancy on NeuroIMAGE ($\rho = 0.5339$, median ratio $= 1.4619$).
- **Physical Cause:** In MINC `minctracc -est_center`, rotations occur about the brain center $\mathbf{c}$ while `transf` is stored at the world origin `(0, 0, 0)`, injecting spurious lever-arm translation $(I - R)\mathbf{c}$ whenever the NIfTI volume center $\mathbf{c} \ne \mathbf{0}$ (NeuroIMAGE offset $54.5\text{ mm}$, Peking $27.2\text{ mm}$, OHSU $24.0\text{ mm}$, KKI $0.0\text{ mm}$).
- **Correction Applied (`3c2b7ac`):** Each scan's `transf` matrix was conjugated to its NIfTI volume geometric center (`motion_target_*_session_1_run1.nii.gz`) via $M_{\text{centered}}(t) = T(-\mathbf{c}) M(t) T(+\mathbf{c})$, raising pooled translation Spearman $\rho$ from $0.8757 \to 0.9704$ (NeuroIMAGE $0.5339 \to 0.8809$).
- **Effect on Blinded Exclusion Counts:** Pooled primary exclusions changed from **39 to 36** (retained: $365 \to 368$; strict exclusions changed from $299 \to 271$, retained $105 \to 133$).
- **Superseded Notice:** All earlier uncentered FD tables recorded in commit `08f87fe` are hereby marked **SUPERSEDED** by the volume-centered counts in commit `3c2b7ac` and [`COHORT_COUNTS_2026-11-18.md`](COHORT_COUNTS_2026-11-18.md).

### 3.3. NeuroIMAGE Registration-Failure Scan (~86° NIAK Divergence)
- One NeuroIMAGE development scan exhibits a NIAK `minctracc` registration failure ($\max |\theta| = 85.98^\circ$ on 203 of 258 retained frames vs. $7.34^\circ$ in Athena `NeuroIMAGE_motion.csv`).
- **Pre-Specified Disposition (No Ad-Hoc Removal):** This scan is **excluded under the primary motion rule** ($N = 368$, automatically exceeding both primary thresholds) and **retained in the co-primary no-exclusion analysis** ($N = 404$).

### 3.4. Strict Motion Threshold (`>0.20 mm` OR `>10%` Frames) — Descriptive Only
- Under the strict sensitivity threshold, **3 of 4 LOIO institution folds drop below 15 subjects per class** (`KKI`: ADHD $= 2$; `NeuroIMAGE`: TDC $= 11$, ADHD $= 8$; `OHSU`: TDC $= 13$, ADHD $= 6$; see [`COHORT_COUNTS_2026-11-18.md`](COHORT_COUNTS_2026-11-18.md)).
- **Protocol Deviation Decision:** The strict threshold is retained for **descriptive attrition reporting only** and is **not run as a LOIO classification model**.

### 3.5. Unified Development ROI Set Definition
- The valid CC200 ROI set is defined **once** across **all 404 development scans** (requiring finite, non-zero temporal variance in 100% of all 404 development scans), and that exact frozen ROI set is used identically by both the primary ($N = 368$) and co-primary no-exclusion ($N = 404$) analyses.
