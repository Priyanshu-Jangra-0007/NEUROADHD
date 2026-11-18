# Framewise Motion Parameter Recovery Investigation Report (Study 2 — Phase 2C)

**Document Type:** Technical & Scientific Investigation Report  
**Branch:** `feat/study2-athena-acquisition`  
**Latest Baseline Commit:** `3b851ad`  
**Registration Reference:** OSF Preregistration ([DOI: 10.17605/OSF.IO/GRBT2](https://doi.org/10.17605/OSF.IO/GRBT2) / [https://osf.io/grbt2](https://osf.io/grbt2))  
**Registered Protocol Hash (SHA-256):** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`  
**Investigation Date:** 2026-11-17  
**Analytical Boundary:** Read-only inspection and remote repository provenance mapping. Zero imaging or functional connectivity data was preprocessed or modeled. NYU remains strictly quarantined. Zero individual subject IDs logged or printed.

---

## 1. Executive Classification & Core Finding

### Formal Classification: **`PARTIAL`**
*(Framewise motion traces identified and located in official Neuro Bureau upstream repository packages on NITRC Release 1600, but not yet downloaded or verified on local disk).*

| Dimension | Current Local Staging | Upstream Repository (NITRC Release 1600) | Compliance / Action Status |
| :--- | :---: | :---: | :--- |
| **Craddock 200 BOLD Time Series** | **`RECOVERED`** | Verified package `3423` (274.4 MB, SHA-256 verified) | Ready for feature extraction. |
| **Motion Summary Extrema** | **`VERIFIED`** | Verified packages `3326`–`3332` (11-column CSVs) | Usable for baseline covariate analysis. |
| **Framewise 6-Parameter Traces** | **`ABSENT`** | Located in `intermediate files` packages (389 MB dev total) | **Requires user-authorized acquisition.** |
| **Power et al. (2012) FD Readiness** | **`NOT READY`** | Feasible upon extraction of framewise `.1D` files | Strictly gated until traces are local. |

---

## 2. Review of Registered Motion Exclusion Requirements

Under Section 4 of the locked OSF preregistration ([PREREGISTRATION.md](PREREGISTRATION.md)):
1. **Framewise Displacement (FD) Metric:** Framewise Displacement must be calculated using the Power et al. (2012) formulation:
   $$\text{FD}_i = |\Delta d_{ix}| + |\Delta d_{iy}| + |\Delta d_{iz}| + |\Delta \alpha_i| + |\Delta \beta_i| + |\Delta \gamma_i|$$
   where $\Delta d_{ix} = d_{(i-1)x} - d_{ix}$, and angular displacements ($\Delta \alpha, \Delta \beta, \Delta \gamma$) in degrees are converted to surface displacement on a sphere of radius $R = 50\text{ mm}$:
   $$|\Delta \alpha_i| = 50\text{ mm} \times \left| \Delta \alpha_i^{\circ} \times \frac{\pi}{180^{\circ}} \right| \approx 0.8727\text{ mm/degree} \times |\Delta \alpha_i^{\circ}|$$
2. **Exclusion Criteria:** A participant is excluded if:
   - $\text{Mean FD} > 0.50\text{ mm}$, OR
   - $> 20\%$ of scanning volumes exhibit $\text{FD}_i > 0.50\text{ mm}$.
3. **Rigorous Methodological Rule:**
   - Both criteria are defined strictly on instantaneous volume-to-volume differences ($\Delta$).
   - A global session maximum displacement relative to the initial reference volume (`Max Motion (mm)`) is mathematically and scientifically distinct from framewise displacement.
   - **Estimating, approximating, or substituting summary maxima for framewise displacement is strictly forbidden.**

---

## 3. Structural Autopsy of Currently Staged Datasets

### 3.1. Primary Time-Series Archive (`ADHD200_CC200_TCs_filtfix.tar.gz`)
* **Inspection Method:** Non-extracting streaming tar reader.
* **Findings:** All 2,416 archive entries were audited. The archive contains:
  - 759 primary band-pass filtered CC200 BOLD time courses (`sfnwmrda...cc200_TCs.1D`).
  - 759 secondary unfiltered CC200 BOLD time courses (`snwmrda...cc200_TCs.1D`).
  - Secondary multi-run time series (`rest_2`, `rest_3`).
  - Site phenotypic CSV files and template masks.
* **Motion Trace Presence:** **0** individual framewise motion traces (`motion.1D` or `rp_*.1D`) are bundled inside the 274.4 MB CC200 archive.

### 3.2. Site Motion Files (`<Site>_motion.csv`)
* **Inspection Method:** Header and tabular distribution analysis across all development sites (KKI, NeuroIMAGE, OHSU, Peking 1, Peking 2, Peking 3).
* **Findings:** Each file consists of an 11-column summary table produced by the Neuro Bureau's `est_motion_afni.pl` script:
  `File`, `Max Motion (mm)`, `Max Motion Time Point`, `Max Rotation (degree)`, `Max Rotation Time Point`, `Max X (mm)`, `Max Y (mm)`, `Max Z (mm)`, `Max Roll (degree)`, `Max Pitch (degree)`, `Max Yaw (degree)`.
* **Row Count:** Exactly 1 row per participant scan session ($N=83$ for KKI, $N=48$ for NeuroIMAGE, $N=237$ for OHSU across runs, $N=85$ for Peking 1, $N=67$ for Peking 2, $N=42$ for Peking 3).
* **Conclusion:** These tables provide **scalar session extrema**, not the instantaneous volume-by-volume time series of the 6 rigid-body parameters.

---

## 4. Investigation of Upstream Neuro Bureau Repositories

### 4.1. Athena Preprocessing Pipeline Architecture
Investigation of the official Athena pipeline source repository (`preprocessed-connectomes-project/adhd200_athena_scripts`) established the exact motion estimation procedure:
1. **Realignment Engine:** AFNI `3dvolreg`.
2. **Command Execution:**
   ```bash
   3dvolreg -prefix rest_volreg -1Dfile motion.1D -dfile motion.dfile rest+orig
   ```
3. **Generated Motion Artifacts:**
   - `motion.1D`: Clean 6-column ASCII text file containing volume-by-volume parameters.
   - Column order: $roll, pitch, yaw$ (degrees counterclockwise), $dS, dL, dP$ (millimeters).
   - Rows: Exactly $T$ lines corresponding to the number of TRs in the scan.
4. **Summary Extraction:** The Neuro Bureau ran `est_motion_afni.pl` on these `motion.1D` files to extract the peak values into `<Site>_motion.csv`, and retained the full `.1D` files in upstream packages.

### 4.2. NITRC Release 1600 Inventory Audit
A comprehensive audit of all 157 file packages hosted under NITRC Release 1600 ([The Neuro Bureau, Group ID: 383](https://www.nitrc.org/frs/?group_id=383&release_id=1600)) revealed the specific distribution tier holding the original framewise traces:

| Package Group | Release Items | Size Range | Contents Description |
| :--- | :--- | :---: | :--- |
| **Tier 1: Time Courses** | `CC200 Time Courses Corrected Filtering` | 274.0 MB | Extracted ROI BOLD time courses only (Already Acquired). |
| **Tier 2: Motion Summaries** | `<Site>_motion.csv` | 4–35 KB each | 11-column scalar extrema tables (Already Acquired). |
| **Tier 3: Intermediate Files** | **`<Site> intermediate files`** | **31–125 MB each** | **Contains per-subject confound regressors, masks, and `motion.1D` / `rp_*.1D` framewise traces.** |
| **Tier 4: Registration Transforms** | `<Site>_preproc_xfms.tar` | 5–11 MB each | Spatial transformation matrices (affine/warp to standard MNI space). |
| **Tier 5: Full Preprocessed** | `<Site> preproc corrected filtering` | 2–4 GB each | Full 4D voxelwise BOLD NIfTI volumes and all derivative streams. |

---

## 5. Identified Source for Framewise Parameter Recovery

The framewise rigid-body motion traces for the development institutions reside in the **`intermediate files`** distribution tier on NITRC Release 1600:

| Development Site | Package Name on NITRC | Download Link ID | Package Size | Expected Framewise Contents |
| :--- | :--- | :---: | :---: | :--- |
| **KKI** | `KKI intermediate files` | `/frs/downloadlink.php/3451` | 124 MB | Per-subject $T=124\text{--}128$ motion traces (`motion.1D`) |
| **NeuroIMAGE** | `Neuroimage intermediate files` | `/frs/downloadlink.php/3812` | 31 MB | Per-subject $T=261$ motion traces (`motion.1D`) |
| **OHSU** | `OHSU intermediate files` | `/frs/downloadlink.php/3563` | 125 MB | Per-subject $T=78\text{--}82$ motion traces (`motion.1D`) |
| **Peking** (1, 2, 3) | `Beijing intermediate files` | `/frs/downloadlink.php/3640` | 109 MB | Per-subject $T=235\text{--}240$ motion traces (`motion.1D`) |
| **Total Dev Acquisition** | — | — | **389 MB** | **All 4 Development Institutions Covered** |

*(Sealed External Site: `NYU intermediate files` exists at package ID `3808` [263 MB], but remains quarantined and is NOT requested for development analysis).*

---

## 6. Expected Physical Properties of Target Traces

When extracted from the intermediate packages, the target files must satisfy the following verified criteria:
1. **Filename Patterns:** `<ID>_rest_1_motion.1D` or `<ID>/session_1/rest_1/motion.1D` (or `rp_<ID>_session_1_rest_1.1D`).
2. **Column Dimensionality:** Exactly 6 numerical columns ($roll, pitch, yaw, dS, dL, dP$).
3. **Row Dimensionality:** Exactly matches the site TR frame counts:
   - KKI: $124\text{--}128$ rows (TR = 2.5s)
   - NeuroIMAGE: $261$ rows (TR = 1.96s)
   - OHSU: $78\text{--}82$ rows (TR = 2.5s)
   - Peking: $235\text{--}240$ rows (TR = 2.0s)
4. **Units:** Rotations in degrees counterclockwise; translations in millimeters.

---

## 7. NYU Sealed-Site Protection Protocol

In accordance with strict study governance:
* The NYU intermediate files package (`NYU intermediate files`, 263 MB) was **not accessed, requested, or downloaded**.
* Zero NYU neural time series or functional connectivity features were inspected.
* NYU remains completely quarantined for locked single-shot evaluation.

---

## 8. Proposed Acquisition & Next Steps

In compliance with instructions (*"Do not download large archives or modify the dataset without reporting the source, expected files and proposed acquisition method first"*):

### Proposed Action Plan
1. **Source Authorization:** Await user authorization to acquire the four development-site intermediate packages from NITRC Release 1600 ($389\text{ MB}$ total):
   - `KKI intermediate files` (124 MB)
   - `Neuroimage intermediate files` (31 MB)
   - `OHSU intermediate files` (125 MB)
   - `Beijing intermediate files` (109 MB)
2. **Acquisition Method:** Download through the user's authenticated NITRC browser session into `NeuroADHD/data/athena/staging/`.
3. **Automated Extraction & Verification:**
   - Extract strictly the `.1D` motion traces into `NeuroADHD/data/athena/motion/`.
   - Verify 6 columns, TR row counts, and physical unit bounds.
   - Run the pre-registered Power et al. (2012) FD calculation script.
   - Mechanically tabulate cohort exclusions in `docs/study2/COHORT_COUNTS_<DATE>.md`.
4. **Protocol Integrity:** The pre-registered protocol, SHA-256 hash, and inclusion thresholds remain 100% frozen.
