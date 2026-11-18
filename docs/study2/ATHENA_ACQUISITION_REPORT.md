# Athena Dataset Acquisition & Verification Report (Study 2 — Phase 2B)

**Document Type:** Formal Acquisition Verification Report  
**Branch:** `feat/study2-athena-acquisition`  
**Registration Reference:** OSF Preregistration ([DOI: 10.17605/OSF.IO/GRBT2](https://doi.org/10.17605/OSF.IO/GRBT2) / [https://osf.io/grbt2](https://osf.io/grbt2))  
**Registered Protocol Hash (SHA-256):** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`  
**Official Release Source:** NITRC Athena Release ([Group ID: 383, Release ID: 1600](https://www.nitrc.org/frs/?group_id=383&release_id=1600))  
**Acquisition Date:** 2026-11-17  
**Data Privacy & Guardrails:** Zero individual subject IDs logged or printed; NYU sealed site maintained under strict analytical quarantine.

---

## 1. Executive Summary: Core Protocol Status

| Protocol Dimension | Verified Status | Governing Scientific Rationale |
| :--- | :---: | :--- |
| **Archive Integrity** | **PASS** | Valid, uncorrupted gzip stream; 2,416 archive entries; 2,404 `.1D` files verified without unpacking to disk. |
| **File Acquisition** | **PASS** | All 8 required files verified present in staging (`NeuroADHD/data/athena/staging/`) with non-zero sizes and locally computed SHA-256 provenance hashes recorded (see Section 2.1). |
| **Motion Summary Metadata** | **VERIFIED** | 11-column per-subject summary tables verified; rotational units in degrees, translational units in mm. |
| **Framewise FD Availability** | **UNRESOLVED** | Available CSV files contain scalar summary extrema only; framewise ($T \times 6$) rigid-body parameter traces (`rp_*.1D`) are absent. |
| **Motion-Based Cohort Exclusion** | **NOT READY** | Strictly gated. Preregistered Power et al. (2012) FD calculation requires framewise traces; calculating/approximating FD from summary maxima is strictly prohibited. |
| **NYU Sealed Protection** | **PASS (QUARANTINED)** | Strict administrative presence verified; zero inspection of NYU neural time series or connectivity matrices. |

---

## 2. Acquired Files & Checksums

All 8 files downloaded from the official NITRC Release 1600 repository and the accompanying phenotypic metadata were verified in the staging directory (`NeuroADHD/data/athena/staging/`):

| # | File Name | Source / Release Item | Size (Bytes) | Human Size | SHA-256 Checksum | Acquisition Status |
| :-: | :--- | :--- | :---: | :---: | :--- | :---: |
| 1 | `ADHD200_CC200_TCs_filtfix.tar.gz` | NITRC Release 1600 "CC200 Time Courses Corrected Filtering" | 287,773,725 | 274.4 MB | `f86a57a660764ac70064f22876eef9df37480b04c1ab0fe192314b86c1bc7359` | **PASS** |
| 2 | `KKI_motion.csv` | NITRC `/frs/downloadlink.php/3326` | 7,517 | 7.3 KB | `1428bc6bc7910623156e52521144c1c8febea0941bea60bf490154467fef5721` | **PASS** |
| 3 | `NeuroIMAGE_motion.csv` | NITRC `/frs/downloadlink.php/3327` | 4,463 | 4.4 KB | `d38c3d3713cbc68f8204b6e388a3b621a2f5f201b8aa889278ed2d7fd7a286d5` | **PASS** |
| 4 | `OHSU_motion.csv` | NITRC `/frs/downloadlink.php/3329` | 20,887 | 20.4 KB | `2cb9c49d0f229550ce5da6c4cb398f85350cd1308d5f7f52759b494c007c67ec` | **PASS** |
| 5 | `Peking_1_motion.csv` | NITRC `/frs/downloadlink.php/3330` | 7,847 | 7.7 KB | `e171b0bb00fb27ce70f54ea0a68866c5c43d17509f2d1a0d32685eec01282713` | **PASS** |
| 6 | `Peking_2_motion.csv` | NITRC `/frs/downloadlink.php/3331` | 6,196 | 6.1 KB | `c7c06bb6277acee521d331e04ad8c15e3c1e17c0bd37723726aa90e1b79693ec` | **PASS** |
| 7 | `Peking_3_motion.csv` | NITRC `/frs/downloadlink.php/3332` | 3,947 | 3.9 KB | `cb6522e182e58ef89825129a31f42c602bf18a8305bfa00fe93da29a80323aae` | **PASS** |
| 8 | `NYU_motion.csv` | NITRC `/frs/downloadlink.php/3328` | 35,609 | 34.8 KB | `15d0287a0a9ef2516f7fed7f6de5b46cb5e54da4c826f1173f2488028d85efb3` | **PASS (Quarantined)** |
| 9 | `adhd200_preprocessed_phenotypics.tsv` | NITRC `/frs/download.php/9024/` | 59,935 | 58.5 KB | `7a37195f0260b04246b833ff4b8050afc4756b8a8f1622feca52944189f5a898` | **PASS** |

### 2.1. SHA-256 Hash Provenance Clarification (Added 2026-11-18)
* **Self-Computed Local Hashes (NITRC Downloads):** NITRC Release 1600 does **not** publish official SHA-256 checksums for these files. Every SHA-256 hash recorded in Section 2 (and for subsequent NITRC staging archives) is a **self-computed local hash** calculated via `hashlib.sha256` / `shasum -a 256` after download to create an immutable provenance record and detect subsequent local corruption or modification. It was **not** compared against a published upstream checksum.
* **Externally Locked Reference Check (`PREREGISTRATION.md`):** By contrast, the SHA-256 verification on `docs/study2/PREREGISTRATION.md` (`e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`) **is** compared against the externally locked reference hash recorded in OSF registrations `grbt2`/`db7nv` and Git commit `2d7ed3b`.

---

## 3. Archive Filename Discrepancy Resolution

* **Observed Discrepancy:** The initial documentation referenced `ADHD200_CC200TCs_filtfix.tar.gz`, while the physical file delivered by NITRC Release 1600 is titled `ADHD200_CC200_TCs_filtfix.tar.gz` (with an underscore between `CC200` and `TCs`).
* **Source Verification:** NITRC Release 1600 file package titled "CC200 Time Courses Corrected Filtering" provides `ADHD200_CC200_TCs_filtfix.tar.gz`.
* **Resolution:** In accordance with instructions, **the archive was not renamed**. Both `scripts/athena_verify_acquisition.py` and `docs/study2/ATHENA_ACQUISITION_MANIFEST.md` were updated to recognize `ADHD200_CC200_TCs_filtfix.tar.gz` as the official primary filename, with `ADHD200_CC200TCs_filtfix.tar.gz` retained as an alias.

---

## 4. Archive Integrity Verification (Non-Extracting Inspection)

Archive integrity was verified directly via streaming gzip/tar reader without unpacking to disk:

* **Archive Status:** Valid, uncorrupted gzip tarball (**`PASS`**).
* **Total Archive Members:** 2,416 entries.
* **Member Composition:**
  - `.1D` Time-Series Files: 2,404 files.
  - Phenotypic / Motion `.csv` Files: 10 files (site-level phenotype tables and `KKI_motion.csv`).
  - Shell Scripts: 1 file (`extract_tcs.sh`, verifying AFNI `3dROIstats` extraction using Craddock `ADHD200_parcellate_200.nii.gz`).
  - Compressed Metadata: 1 `.gz` template mask.
* **Top-Level Site Directories Discovered in Archive:**
  - `NYU`: 799 files (quarantined)
  - `OHSU`: 475 files
  - `WashU`: 305 files (excluded a priori)
  - `Pittsburgh`: 179 files (excluded a priori)
  - `Peking_1`: 171 files
  - `KKI`: 168 files
  - `Peking_2`: 135 files
  - `NeuroIMAGE`: 97 files
  - `Peking_3`: 85 files
* **Time-Series Filename Patterns (IDs Masked):**
  - `sfnwmrda<ID>_session_1_rest_1_cc200_TCs.1D` ($N=759$): Primary band-pass filtered ($0.009\text{--}0.08\text{ Hz}$), nuisance-regressed, non-GSR `rest_1` time courses across the consortium.
  - `snwmrda<ID>_session_1_rest_1_cc200_TCs.1D` ($N=759$): Secondary unfiltered series.
  - Multi-run files for sites with multiple acquisitions (e.g. `rest_2`, `rest_3`).

---

## 5. Motion Metadata: Summary vs. Framewise Parameters

### 5.1. Nature of the Available Motion CSV Files
Inspection of `KKI_motion.csv`, `NeuroIMAGE_motion.csv`, `OHSU_motion.csv`, `Peking_1_motion.csv`, `Peking_2_motion.csv`, and `Peking_3_motion.csv` revealed that these files are **per-subject scalar extrema summary tables** produced by the Neuro Bureau's `est_motion_afni.pl` script.

Each row corresponds to one scan session and contains exactly 11 summary columns:
1. `File`: Target realignment parameter trace (e.g. `rp_<ID>_session_1_rest_1.1D`).
2. `Max Motion (mm)`: Peak translational head displacement across the entire scan.
3. `Max Motion Time Point`: TR index where peak displacement occurred.
4. ` Max Rotation (degree)`: Peak angular rotation across the entire scan.
5. ` Max Rotation Time Point`: TR index where peak rotation occurred.
6. ` Max X (mm)`: Peak displacement along Left-Right axis.
7. ` Max Y (mm)`: Peak displacement along Anterior-Posterior axis.
8. ` Max Z (mm)`: Peak displacement along Superior-Inferior axis.
9. ` Max Roll (degree)`: Peak rotation about Inferior-Superior axis.
10. ` Max Pitch (degree)`: Peak rotation about Left-Right axis.
11. ` Max Yaw (degree)`: Peak rotation about Anterior-Posterior axis.

### 5.2. Scientific Insufficiency for Power et al. (2012) FD
> [!IMPORTANT]
> **Scientific Finding:** The available motion CSVs contain summary maxima, **not** frame-by-frame realignment time series.
> 
> * **Power et al. (2012) Formulation:** Framewise Displacement at volume $i$ is defined by backward differences between consecutive time points:
>   $$\text{FD}_i = |\Delta d_{ix}| + |\Delta d_{iy}| + |\Delta d_{iz}| + |\Delta \alpha_i| + |\Delta \beta_i| + |\Delta \gamma_i|$$
>   where $\Delta d_{ix} = d_{(i-1)x} - d_{ix}$.
> * **Exclusion Criteria Dependency:** Evaluating whether $\text{Mean FD} > 0.50\text{ mm}$ or whether $> 20\%$ of frames exceed $0.50\text{ mm}$ strictly requires the instantaneous volume-by-volume $\text{FD}_i$ time series ($T \approx 120\text{--}260$ frames per participant).
> * **Prohibition on Heuristic Estimation:** A global scalar maximum displacement relative to reference (`Max Motion (mm)`) does **not** equal framewise displacement ($\Delta$). Under our preregistered protocol, calculating, estimating, or approximating FD from maximum scalar values is **strictly forbidden**.

### 5.3. Audit of Downloaded Archive for Framewise Parameters
A full inspection of the 2,416 members of `ADHD200_CC200_TCs_filtfix.tar.gz` was conducted:
- The archive contains exclusively Craddock 200 BOLD time courses (`sfnwmrda...` and `snwmrda...`), template masks, and site phenotype CSVs.
- The individual 6-parameter frame-by-frame motion traces (`rp_<ID>_session_1_rest_1.1D` or `<ID>_rest_1_motion.1D`) are **NOT present** in the 274 MB CC200 time-series archive.

### 5.4. Protocol Dependency Status: UNRESOLVED
* **Motion Summary Metadata:** **`VERIFIED`** (Physical units confirmed: degrees and mm).
* **Framewise FD Availability:** **`UNRESOLVED`** (Frame-by-frame 6-parameter traces are unavailable in the current download set).
* **Motion-Based Cohort Exclusion:** **`NOT READY until valid FD data are available`**.

---

## 6. NYU Quarantine Status & Member-Read Audit (Updated 2026-11-18)

The NYU external validation dataset ($N = 216$ `rest_1` scans) remains protected under strict scientific quarantine:
* **Administrative Presence Check:** `NYU_motion.csv` verified present (35,609 bytes, self-computed SHA-256: `15d0287a0a9ef2516f7fed7f6de5b46cb5e54da4c826f1173f2488028d85efb3`).
* **Archive Presence Check:** 799 entries in `NYU/` subdirectory of `ADHD200_CC200_TCs_filtfix.tar.gz` confirmed.
* **CC200 Archive NYU Extraction & Read Audit:**
  - **Zero NYU members have been extracted to disk.** No file from `ADHD200_CC200_TCs_filtfix.tar.gz` has been unpacked to disk.
  - **Zero NYU numerical BOLD values, time series, connectivity matrices, correlations, or diagnosis labels have ever been parsed, printed, stored, or analyzed.**
  - **Full Transparency Note:** During Phase 2B acquisition checks, only member names and counts were read. During Phase 2D's in-memory frame-count check, an unfiltered Python loop over `sfnwmrda...rest_1` members called `tf.extractfile(m)` and counted non-empty lines (`len(lines)`) across all sites including the 216 `NYU/` `rest_1` members, returning only the aggregate site count (`nyu: 216`) in memory without parsing, storing, or inspecting any BOLD signal values or individual IDs.
  - **Strict Rule Going Forward:** All future CC200 archive reads and extractions will explicitly filter to **development-site members only** (`KKI`, `NeuroIMAGE`, `OHSU`, `Peking_1`, `Peking_2`, `Peking_3`) and skip `NYU/` entirely (both on disk and in memory).
* **Analytical Quarantine Enforcement:**
  - Zero inspection of NYU BOLD time-series values.
  - Zero functional connectivity feature extraction.
  - Zero correlation matrices computed or visualized.
  - Zero model tuning or feature selection using NYU data.

---

## 7. Execution Boundary Adherence

In strict compliance with instructions:
- The archive was **NOT extracted** to disk.
- Cohort-wide Framewise Displacement (FD) was **NOT calculated**.
- No cohort counts were finalized or committed.
- No preprocessing or machine learning model training was performed.
- Zero individual subject IDs were printed or logged.

---

## 8. Summary of Protocol Dependencies & Next Steps

1. **Acquired & Verified:**
   - Primary CC200 resting-state BOLD time series archive (`ADHD200_CC200_TCs_filtfix.tar.gz`) is verified and intact.
   - Phenotypic metadata table (`adhd200_preprocessed_phenotypics.tsv`) is verified and intact.
   - Motion summary extrema tables for all sites are verified and intact.
2. **Unresolved Protocol Dependency:**
   - The framewise 6-parameter rigid-body motion traces (`rp_*.1D`) must be sourced from the verified Neuro Bureau site packages before the preregistered Power et al. (2012) motion-based exclusion rule can be executed.
3. **Status:**
   - Archive integrity: **PASS**
   - File acquisition: **PASS**
   - Motion summary metadata: **VERIFIED**
   - Framewise FD availability: **UNRESOLVED**
   - Motion-based cohort exclusion: **NOT READY**
