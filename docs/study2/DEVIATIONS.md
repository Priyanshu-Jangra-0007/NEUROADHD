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
  4. Report the pre-registered **no-exclusion analysis** ($N = 404$, retaining all scans without FD-based exclusion) as the registered **Section 5.4 sensitivity analysis** alongside the **primary analysis** (pooled LOIO AUC at $N = 368$ after the registered Section 5.3 primary exclusion rule; Section 7.1's $N = 404$ is the preliminary pre-exclusion count from Section 4.2). *(Wording updated 2026-11-18 prior to any real-label analysis; see Section 4.1.)*

---

## 2. Motion-Only Baseline & Label Blinding Rules

1. **Motion-Only Baseline Parity:** The pre-registered motion-only baseline model uses **whichever motion source is selected by the rule above** — Branch (a) AFNI `rp_*.1D` if all 404 exist, or Branch (b) NIAK-derived parameters (first 3 frames removed) otherwise.
2. **Strict Blinding to Diagnosis Labels:** **No FD computation, threshold check, alignment check, or fallback decision uses or joins diagnosis labels (`DX`).** All motion extraction and FD thresholding are executed strictly blind to diagnostic class.
3. **NYU Quarantine:** This rule applies exclusively to development-site scans (KKI, NeuroIMAGE, OHSU, Peking 1–3). No NYU archive is opened, extracted, or analyzed during development.

---

## 3. Dated Deviation & Implementation Log — 2026-11-18

### 3.1. Activation of Fallback Branch (b)
- All six development `<Site>_preproc_xfms.tar` archives contained FSL FNIRT structural-to-template warp fields only (`0 / 404` AFNI `rp_*.1D` traces; commit `08f87fe`).
- Per Section 1 above, **Branch (b)** was activated: NIAK-derived Power et al. (2012) FD with the first 3 frames removed, labelled a protocol deviation, with the primary analysis being pooled LOIO AUC at $N = 368$ after the registered Section 5.3 primary exclusion rule and the no-exclusion analysis at $N = 404$ being the registered Section 5.4 sensitivity analysis.

### 3.2. FD Calculator Correction: NIfTI Volume-Centre Conjugation (Pre-Label-Join)
- **Discovery (Strictly Label-Free):** During label-free validation of NIAK-derived motion against the official Athena `*_motion.csv` extrema ([`NIAK_VS_ATHENA_MOTION_VALIDATION.md`](NIAK_VS_ATHENA_MOTION_VALIDATION.md)), executed **before any diagnosis labels were joined**, naive extraction of world-origin translation `transf[:3, 3]` exhibited a severe translation discrepancy on NeuroIMAGE ($\rho = 0.5339$, median ratio $= 1.4619$).
- **Physical Cause:** In MINC `minctracc -est_center`, rotations occur about the brain center $\mathbf{c}$ while `transf` is stored at the world origin `(0, 0, 0)`, injecting spurious lever-arm translation $(I - R)\mathbf{c}$ whenever the NIfTI volume center $\mathbf{c} \ne \mathbf{0}$ (NeuroIMAGE offset $54.5\text{ mm}$, Peking $27.2\text{ mm}$, OHSU $24.0\text{ mm}$, KKI $0.0\text{ mm}$).
- **Correction Applied (`3c2b7ac`):** Each scan's `transf` matrix was conjugated to its NIfTI volume geometric center (`motion_target_*_session_1_run1.nii.gz`) via $M_{\text{centered}}(t) = T(-\mathbf{c}) M(t) T(+\mathbf{c})$, raising pooled translation Spearman $\rho$ from $0.8757 \to 0.9704$ (NeuroIMAGE $0.5339 \to 0.8809$).
- **Effect on Blinded Exclusion Counts:** Pooled primary exclusions changed from **39 to 36** (retained: $365 \to 368$; strict exclusions changed from $299 \to 271$, retained $105 \to 133$).
- **Superseded Notice:** All earlier uncentered FD tables recorded in commit `08f87fe` are hereby marked **SUPERSEDED** by the volume-centered counts in commit `3c2b7ac` and [`COHORT_COUNTS_2026-11-18.md`](COHORT_COUNTS_2026-11-18.md).

### 3.3. NeuroIMAGE Registration-Failure Scan (~86° NIAK Divergence)
- One NeuroIMAGE development scan exhibits a NIAK `minctracc` registration failure ($\max |\theta| = 85.98^\circ$ on 203 of 258 retained frames vs. $7.34^\circ$ in Athena `NeuroIMAGE_motion.csv`).
- **Pre-Specified Disposition (No Ad-Hoc Removal):** This scan is **excluded under the primary motion rule** ($N = 368$, automatically exceeding both primary thresholds) and **retained in the registered Section 5.4 no-exclusion sensitivity analysis** ($N = 404$).

### 3.4. Strict Motion Threshold (`>0.20 mm` OR `>10%` Frames) — Descriptive Only
- Under the strict sensitivity threshold, **3 of 4 LOIO institution folds drop below 15 subjects per class** (`KKI`: ADHD $= 2$; `NeuroIMAGE`: TDC $= 11$, ADHD $= 8$; `OHSU`: TDC $= 13$, ADHD $= 6$; see [`COHORT_COUNTS_2026-11-18.md`](COHORT_COUNTS_2026-11-18.md)).
- **Protocol Deviation Decision:** The strict threshold is retained for **descriptive attrition reporting only** and is **not run as a LOIO classification model**.

### 3.5. Unified Development ROI Set Definition
- The valid CC200 ROI set is defined **once** across **all 404 development scans** (requiring finite, non-zero temporal variance in 100% of all 404 development scans), and that exact frozen ROI set is used identically by both the primary analysis ($N = 368$) and the registered Section 5.4 no-exclusion sensitivity analysis ($N = 404$).

---

## 4. Dated Wording Fix & Clarifications — 2026-11-18 (Pre-Real-Label Analysis)

### 4.1. Wording Fix: Primary Analysis vs. Section 5.4 Sensitivity Analysis
- **Wording Correction (Fixed Before Any Real-Label Analysis):** Earlier wording in `DEVIATIONS.md` (commit `ac517ce`) described the no-exclusion analysis as "co-primary." That wording is corrected here without rewriting Git history and **strictly before any real-label analysis has been run**:
  - The **primary analysis** is **pooled LOIO ROC-AUC at $N = 368$** after applying the registered primary motion exclusion rule (`PREREGISTRATION.md`, Section 5.3).
  - The **no-exclusion analysis at $N = 404$** is the registered **Section 5.4 sensitivity analysis** (`PREREGISTRATION.md`, Section 5.4).
  - The figure $N = 404$ cited in `PREREGISTRATION.md` Section 7.1 is the **preliminary pre-exclusion count** from Section 4.2.

### 4.2. Clarifications, Not Deviations (2026-11-18)
The following implementation details clarify operational choices that conform to `PREREGISTRATION.md` and are **not protocol deviations**:

- **(a) Pearson $r$ Clipping Before Fisher $z$:** Off-diagonal Pearson correlations are clipped to `[-1 + 1e-7, 1 - 1e-7]` (`FISHER_CLIP_EPS = 1e-7` in `scripts/extract_study2_dev_features.py`) prior to $\operatorname{arctanh}(r)$ strictly for numerical stability to prevent $\pm\infty$ if $|r| = 1$.
- **(b) Why 190 ROIs Are Kept When `.1D` Headers Span `Mean_1..Mean_200`:** Direct inspection of the official Athena atlas mask inside `ADHD200_CC200_TCs_filtfix.tar.gz` (`templates/ADHD200_parcellate_200.nii.gz`, dimensions $49 \times 58 \times 47$) confirms it contains exactly **190 populated non-zero integer ROI labels** spanning `[1, 200]` — **10 cluster IDs** (`2, 5, 11, 25, 34, 68, 82, 95, 126, 191`) have 0 voxels in Craddock's 200-cluster normalized-cut parcellation and are absent from `3dROIstats` output. Consequently, every `.1D` file has **192 columns** (`File`, `Sub-brick`, plus **190** `Mean_<id>` columns), and **190 / 190** ROIs are kept (**0 dropped**), yielding $P = 190 \times 189 / 2 = 17,955$ upper-triangle edges.
- **(c) Equivalence of the ROI Set Across $N = 404$ and $N = 368$:** The valid ROI set was defined from all **404** development scans (finite, $\text{std} > 0$ in 100% of scans; minimum temporal $\text{std} = 0.004031 > 0$). Because **0 ROIs were dropped** across all 404 scans, the resulting 190-ROI set is **mathematically identical** to the ROI set obtained on the $N = 368$ primary subset.
- **(d) Exact Definition of "Max Displacement" Used by the Scripts (Quoted Verbatim from Code):**
  In `scripts/compute_athena_fd.py` (lines 292–294), computed from the volume-centered, first-3-frames-dropped NIAK translations `params[:, 3:6]` (mm):
  ```python
  # Relative translation from first retained frame (Euclidean mm)
  rel_trans = params[:, 3:6] - params[0, 3:6]
  max_disp_mm = float(np.max(np.linalg.norm(rel_trans, axis=1)))
  ```
  And passed as the second feature of `motion_X` (`[mean_fd_mm, max_disp_mm]`) in `scripts/extract_study2_dev_features.py` (lines 257–258):
  ```python
  motion_X[idx, 0] = float(m_row["mean_fd_mm"])
  motion_X[idx, 1] = float(m_row["max_disp_mm"])
  ```

### 4.3. Explicit Section 5.2 Source-Substitution Statement (2026-11-18, Pre-Run)
- **Protocol Deviation (Registration Section 5.2):** Per Section 1 Branch (b), Framewise Displacement (FD) for the development cohort was computed from the Neuro Bureau **NIAK-derived $4 \times 4 \times T$ rigid-body transforms** (volume-centered, first 3 frames removed) rather than the Athena **AFNI `3dvolreg` `rp_*.1D` files** specified in `PREREGISTRATION.md` Section 5.2, because the six development `<Site>_preproc_xfms.tar` archives contained `0 / 404` `rp_*.1D` files. This entry is committed prior to executing the real-label development analysis.

### 4.4. Post-Freeze Acquisition of `intermediate_nyu.tar` (2026-11-18, Pre-Sealed-Run)
- **Planned Sealed-Site Motion Acquisition:** As pre-specified in [`SEALED_RUN_PLAN.md`](SEALED_RUN_PLAN.md) (`837b33c`), `intermediate_nyu.tar` (NITRC Release 1600 `/frs/downloadlink.php/3808`) was downloaded **after** the sealed code freeze (`study2-sealed-frozen`, `a75d6b4`) and copied to Git-ignored `NeuroADHD/data/athena/staging/intermediate_nyu.tar`.
- **File Provenance:** Size = `275,527,680` bytes (`262.76 MB`); self-computed SHA-256 = `9640ee0a1623efe91d9eab9ad6a4024d240cdfad3886ecf9b1124a778b141eb4`; non-extracting TAR integrity = `PASS` (`2,665` members, `1,368` files).
- **Quarantine Confirmation:** Zero NYU imaging, time-series, or motion/FD values were inspected or computed prior to the single sealed run. Frozen code and tag `study2-sealed-frozen` remain untouched.

### 4.5. Frame-Drop Count & `.1D` Header-Row Clarification (2026-11-18, Post-Run Clarification — No Code Change)
- **Actual Frames Dropped by Frozen FD Code:** `scripts/compute_athena_fd.py` (`DUMMY_FRAMES_DROPPED = 3`, line 42; also imported by `scripts/run_study2_sealed_site.py`, line 236) slices `transf[:, :, 3:]`, dropping the **first 3 frames** of each NIAK $4 \times 4 \times T_{\text{NIAK}}$ rigid-body transform array and retaining **$T_{\text{NIAK}} - 3$** motion frames for FD and `max_disp_mm`.
- **Relation to the `.1D` Header Row (`3` vs. `4` Frames):** In `scripts/compute_athena_fd.py` (`load_dev_cc200_frame_counts`, line 207), `n_frames = sum(1 for line in stream if line.strip())` counted **all non-empty lines** in each Athena CC200 `.1D` file, **including the 1 header row** (`File`, `Sub-brick`, `Mean_<id>...`). As verified across all `404` development scans in `scripts/extract_study2_dev_features.py` (lines 202–207, `ts.shape[0] + 1 == frames_cc200`, `0` mismatches), each `.1D` file contains **$T_{\text{BOLD}} = T_{\text{NIAK}} - 4$ BOLD data rows** (matching Athena's 4-volume removal `3dcalc '[4..$]'`) plus **1 header row** ($T_{\text{BOLD}} + 1 = T_{\text{NIAK}} - 3$ total non-empty lines).
- **Summary of Alignment:** Relative to the raw NIAK transforms ($T_{\text{NIAK}}$), Athena CC200 `.1D` BOLD time series contain **4 fewer data rows** ($T_{\text{BOLD}} = T_{\text{NIAK}} - 4$), while the frozen FD calculator dropped **3 leading NIAK frames** (retaining $T_{\text{NIAK}} - 3 = T_{\text{BOLD}} + 1$ motion frames, i.e., 1 extra leading NIAK frame at index 3 with $\text{FD}_0 = 0.0$ immediately preceding the $T_{\text{BOLD}}$ BOLD data rows). Code and recorded results are not modified.
