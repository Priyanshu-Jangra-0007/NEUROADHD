# Framewise Motion Trace Validation Report (Study 2 — Phase 2D)

**Document Type:** Technical & Scientific Validation Report  
**Branch:** `feat/study2-athena-acquisition`  
**Registration Reference:** OSF Preregistration ([DOI: 10.17605/OSF.IO/GRBT2](https://doi.org/10.17605/OSF.IO/GRBT2) / [https://osf.io/grbt2](https://osf.io/grbt2))  
**Registered Protocol Hash (SHA-256):** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`  
**Inspection Date:** 2026-11-17  
**Analytical Boundary:** Non-extracting, in-memory stream inspection of staged intermediate archives. Zero files extracted to disk. Zero imaging or connectivity features preprocessed or modeled. Zero framewise displacement calculations executed. Zero participant exclusions applied. NYU hold-out remains strictly quarantined. Zero individual subject identifiers logged or printed.

---

## 1. Executive Summary & Formal Classification

### Formal Classification: **`PARTIAL`**
*(Volume-by-volume 3D rigid-body motion transformation traces identified, verified, and mapped 100% 1-to-1 to all 404 development subjects; however, pipeline provenance [NIAK Octave affine matrices vs. Athena AFNI realignment traces] and a systematic 3-frame dummy scan offset represent unresolved protocol dependencies).*

| Validation Dimension | Status | Detailed Finding |
| :--- | :---: | :--- |
| **Archive Integrity** | **PASS** | All 4 development intermediate tarballs verified without extraction (4,274 members, 1,846 files, ~389.1 MB). |
| **Trace Availability** | **FOUND** | Per-subject volume-by-volume rigid-body motion traces exist (`motion_parameters_...mat`). |
| **Format & Representation** | **AFFINE 3D** | Octave/MATLAB 3D matrix `transf` ($4 \times 4 \times T$), containing 3D rotations and 3D translations. |
| **Subject Mapping (rest_1)** | **100% (404/404)** | Exact 1:1 match across all development cohorts (KKI 83, NeuroIMAGE 48, OHSU 79, Peking 194). 0 missing, 0 extra. |
| **Volume Alignment** | **$T - 3$ OFFSET** | CC200 BOLD time series are systematically shorter by exactly 3 TRs (dummy scan equilibration drop). |
| **Pipeline Provenance** | **UNRESOLVED** | Traces originate from the NIAK (McGill CBRAIN/Octave) pipeline, not the Athena (AFNI/FSL) pipeline. |
| **FD Readiness** | **GATED** | Requires formal methodological protocol sign-off before extraction or threshold filtering. |

---

## 2. Archive Inspection & Candidate Trace Identification

Streaming, non-extracting inspection was performed on all four development archives staged in `NeuroADHD/data/athena/staging/`:

```
NeuroADHD/data/athena/staging/
├── intermediate_kki.tar         (130,150,400 bytes, 749 members, 250 files)
├── intermediate_neuroimage.tar   (32,501,760 bytes, 433 members, 144 files)
├── intermediate_ohsu.tar        (131,348,480 bytes, 1,345 members, 870 files)
└── intermediate_peking.tar      (114,042,880 bytes, 1,747 members, 582 files)
```

### 2.1. Member Pattern Inventory

No flat ASCII 6-column motion parameter files (such as `motion.1D` or `rp_*.1D`) were identified in any of the four intermediate archives. Instead, the archives contain a standardized tripartite set of intermediate motion outputs per scanning run:

1. **`motion_parameters_X_<ID>_session_<S>_run<R>.mat`**:
   Full volume-by-volume rigid-body transformation time series in Octave text format.
2. **`motion_Wrun_X_<ID>_session_<S>_run<R>.mat`**:
   Within-run transformation matrices (identical matrix series to `motion_parameters_*.mat`, differing only by file generation timestamp).
3. **`motion_target_X_<ID>_session_<S>_run<R>.nii.gz`**:
   3D reference target volume utilized for volume realignment.
4. **Site Phenotypic Tables**:
   - `intermediate_kki/KKI_phenotypic.csv`
   - `intermediate_ohsu/OHSU_phenotypic.csv`
5. **Cross-Session Matrices (OHSU only)**:
   - `motion_Wsession_X_<ID>_session_<S>_run<R>.mat` (4x4 cross-run registration matrices).

---

## 3. Structural & Mathematical Verification of Candidate Traces

Representative files from each site were inspected directly from the uncompressed stream without touching the filesystem.

### 3.1. Data Format & Variable Definition

Each `motion_parameters_...mat` file is stored as an Octave 3.2.4 text-formatted matrix:
- **Origin Header:** `# Created by Octave 3.2.4 ... <cbrainsys@mindstorm.cbrain.mcgill.ca>`
- **Variable Name:** `transf`
- **Data Type:** `matrix`
- **Dimensions:** 3 dimensions: `4 4 T`, where $T$ is the number of time points (volumes) in the scan.

### 3.2. Geometric Rigidity & Orthogonality Verification

In homogeneous coordinates, every temporal slice $t \in [1, T]$ represents an affine spatial transformation:
$$M(t) = \begin{pmatrix} R(t) & \mathbf{t}(t) \\ \mathbf{0}^T & 1 \end{pmatrix} \in SE(3)$$

Mathematical verification across representative samples from all four development sites confirmed:
1. **Determinant:** $\det(R(t)) = 1.000000$ across all time frames.
2. **Orthogonality:** Maximum deviation from orthogonality $\| R^T R - I \|_\infty \le 1.7 \times 10^{-7}$ (within single-precision floating-point bounds).
3. **Bottom Row:** Exactly $[0, 0, 0, 1]$ across all frames.
4. **Conclusion:** Every temporal slice is a mathematically pure, unscaled 3D rigid-body Euclidean transformation matrix.

### 3.3. Parameter Extraction & Physical Units

From the transformation matrix $M(t)$:
- **Translations:** $\mathbf{t}(t) = (t_x, t_y, t_z)^T$ are stored in the 4th column (indices $[0:3, 3]$). The observed dynamic ranges (e.g. $[-0.25, 0.47]\text{ mm}$ for KKI; $[-0.83, 2.03]\text{ mm}$ for NeuroIMAGE) conform to displacement in **millimeters (mm)**.
- **Rotations:** The $3 \times 3$ submatrix $R(t)$ encodes 3D angular orientation relative to the target run volume. Decomposition into standard Euler/Tait-Bryan angles (roll $\alpha$, pitch $\beta$, yaw $\gamma$) via:
  $$\beta = \arcsin(-R_{31}), \quad \alpha = \text{atan2}(R_{32}, R_{33}), \quad \gamma = \text{atan2}(R_{21}, R_{11})$$
  yields rotation angles in **radians**, which convert to degrees ($\times \frac{180}{\pi}$). The resulting angular ranges (e.g., $[-1.12^\circ, 0.53^\circ]$ for KKI; $[-3.71^\circ, 3.62^\circ]$ for NeuroIMAGE) match physiological resting-state motion.

---

## 4. Cohort Mapping & Frame Count Alignment

### 4.1. Subject Mapping to CC200 `rest_1`

Cross-referencing the candidate intermediate files against the pre-filtered CC200 BOLD time courses (`ADHD200_CC200_TCs_filtfix.tar.gz`) demonstrates **100% cohort completeness**:

| Development Site | CC200 `rest_1` Scans | Candidate Motion Traces | Missing Traces | Extra Scans | Match Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KKI** | 83 | 83 | 0 | 0 | **100% EXACT** |
| **NeuroIMAGE** | 48 | 48 | 0 | 0 | **100% EXACT** |
| **OHSU** (`rest_1`) | 79 | 79 | 0 | 0 | **100% EXACT** |
| **Peking** (Sites 1, 2, 3) | 194 | 194 | 0 | 0 | **100% EXACT** |
| **Total Development Cohort** | **404** | **404** | **0** | **0** | **100% EXACT** |

*(Note: Multi-run OHSU scans include 78 `run2`, 78 `run3`, and 2 `run4` traces; only `run1` is selected for `rest_1` analysis).*

### 4.2. Frame Count Discrepancy & Systematic Offset

A systematic difference in time-series length was identified across all 404 participants:

| Site | Raw Intermediate Frames ($T_{\text{raw}}$) | CC200 BOLD Frames ($T_{\text{CC200}}$) | Offset ($\Delta T$) | Participant Breakdown |
| :--- | :---: | :---: | :---: | :--- |
| **KKI** | 124 or 152 | 121 or 149 | **-3** | $N=54$ at $121/124$; $N=29$ at $149/152$ |
| **NeuroIMAGE** | 261 | 258 | **-3** | $N=48$ at $258/261$ |
| **OHSU** | 78 or 76 | 75 or 73 | **-3** | $N=78$ at $75/78$; $N=1$ at $73/76$ |
| **Peking** | 236 | 233 | **-3** | $N=194$ at $233/236$ |

**Scientific Etiology of Offset:**
In the Athena AFNI/FSL pipeline, the first 3 or 4 volumes of each functional run were removed prior to band-pass filtering and nuisance regression to allow for T1 magnetic saturation equilibration (dummy scan removal). The intermediate NIAK packages, however, preserved the raw scanner volume series starting from volume 1.

---

## 5. Summary Statistics vs. Framewise Traces

The relationship between the staged datasets is clarified below:

| Feature Dimension | `<Site>_motion.csv` (Athena Summary) | `motion_parameters_...mat` (Intermediate) |
| :--- | :--- | :--- |
| **Data Nature** | Scalar extrema table (1 row / participant) | 3D array time series ($T$ matrices / participant) |
| **Information Content** | Global maximum translation/rotation across scan | Complete frame-by-frame 6D trajectory ($SE(3)$) |
| **Derivation Tool** | Neuro Bureau `est_motion_afni.pl` on AFNI `rp_*.1D` | McGill CBRAIN / Octave NIAK `motion_correction` |
| **Power et al. (2012) FD** | **IMPOSSIBLE** (cannot compute volume $\Delta$) | **FEASIBLE** (exact volume-to-volume $\Delta$ derivable) |

---

## 6. Critical Scientific & Protocol Issues (The Blocker)

Although volume-by-volume motion parameters exist for all 404 development subjects, proceeding immediately to exclusion is blocked by two methodological discrepancies:

### 1. Cross-Pipeline Origin (NIAK vs. Athena)
- The BOLD connectivity features are extracted from the **Athena** pipeline (AFNI `3dvolreg` alignment).
- The intermediate motion traces were generated by the **NIAK** pipeline (MINC/Octave rigid-body alignment to run median volume).
- While head motion is a physical phenomenon of the participant, slight algorithmic differences exist between AFNI's Levenberg-Marquardt optimizer and NIAK's iterative intensity-based registration.

### 2. Dummy Scan Truncation Alignment
- Calculating Power et al. (2012) Framewise Displacement on the $T_{\text{raw}}$ series yields $T_{\text{raw}} - 1$ displacement values.
- To align with the $T_{\text{raw}} - 3$ BOLD frames in CC200, the first 3 frames must be truncated.
- The preregistered protocol ([PREREGISTRATION.md](PREREGISTRATION.md)) states that subject exclusion is evaluated over all analyzed scanning volumes; applying FD filtering requires establishing whether the criterion ($>20\%$ of volumes with $\text{FD} > 0.50\text{ mm}$) applies to the full scan ($T_{\text{raw}}$) or the trimmed analysis window ($T_{\text{CC200}}$).

---

## 7. Recommended Action Paths

Three potential options are available to maintain compliance with open-science standards:

1. **Option A (Pragmatic Mathematical Derivation):**
   - Use the verified $4 \times 4 \times T$ transformation matrices in `intermediate_*.tar`.
   - Decompose into 6 standard realignment parameters ($t_x, t_y, t_z, \text{roll}, \text{pitch}, \text{yaw}$).
   - Discard the first 3 dummy volumes to match the exact temporal window of the CC200 BOLD time courses.
   - Calculate Power et al. (2012) FD and apply preregistered exclusion thresholds.
   - *Advantage:* 100% complete data already staged; uses actual volume-by-volume kinematic traces.
2. **Option B (Strict AFNI Trace Recovery):**
   - Search upstream Neuro Bureau archives for the specific AFNI `rp_*.1D` outputs from the Athena pipeline.
   - *Limitation:* May require downloading multi-gigabyte full preprocessing archives (Tier 5: 2–4 GB per site) if separate `.1D` packages do not exist.
3. **Option C (Preregistration Note / Addendum):**
   - Record an explicit methodological protocol memo documenting that framewise head kinematics are derived from the official Neuro Bureau intermediate alignment matrices.

---

## 8. Analytical Boundary Conformance

- **NYU Quarantine:** Intact. Zero NYU intermediate packages downloaded or analyzed.
- **Protocol Immutability:** [docs/study2/PREREGISTRATION.md](PREREGISTRATION.md) SHA-256 hash verified unchanged: `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`.
- **Subject Privacy:** Zero individual participant IDs logged or printed in this report.
- **Git State:** Working tree clean except for untracked validation documentation; zero commits or pushes executed.
