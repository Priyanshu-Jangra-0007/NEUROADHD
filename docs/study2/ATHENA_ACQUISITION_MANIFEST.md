# Athena Dataset Acquisition & Ingestion Manifest (Study 2)

**Document Type:** Dataset Acquisition & Readiness Manifest  
**Branch:** `feat/study2-athena-acquisition`  
**Registration Reference:** OSF Preregistration ([DOI: 10.17605/OSF.IO/GRBT2](https://doi.org/10.17605/OSF.IO/GRBT2) / [https://osf.io/grbt2](https://osf.io/grbt2))  
**Registered Protocol Hash (SHA-256):** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`  
**Execution Boundary:** Administrative research and acquisition readiness only. Zero data is downloaded, processed, or modeled during this phase.

---

## 1. Registered Requirements Summary

All requirements in this section are extracted directly from the binding OSF pre-registration protocol ([docs/study2/PREREGISTRATION.md](PREREGISTRATION.md)):

* **Dataset Source & Pipeline:** Pre-computed resting-state fMRI derivatives from the **Athena pipeline** (developed by Cameron Craddock / The Neuro Bureau using AFNI and FSL).
* **Primary Preprocessing Variant:** 200-region functional parcellation, band-pass filtered, without global signal regression (non-GSR), acquisition run 1 (`rest_1`).
* **Fallback Variant Rule:** If band-pass filtered non-GSR 200-region files are not available, use the only available 200-region variant. All other atlas scales (AAL, CC400), GSR-cleaned series, or alternate runs (`rest_2`) are designated strictly exploratory.
* **Feature Extraction:** Pairwise Pearson correlation followed by Fisher $z$-transformation ($z = \frac{1}{2} \ln \frac{1+r}{1-r}$) across valid ROIs (upper triangle).
* **ROI Ingestion Rule:** Valid ROI set defined **strictly from the 4 development institutions** (KKI, NeuroIMAGE, OHSU, Peking). An ROI must exhibit non-zero variance across 100% of analyzed development subjects. If an ROI is missing or constant in the sealed site (NYU), its correlation edges in NYU are imputed with zeros ($r=0, z=0$); development feature selection and weights remain unaltered.
* **Head Motion Metric & Exclusion:** Framewise Displacement calculated via Power et al. (2012) formula using verified 6-parameter motion traces ($roll, pitch, yaw$ in degrees converted to arc-length on a 50mm sphere; $dS, dL, dP$ in mm). Primary exclusion threshold: $\text{Mean FD} > 0.50\,\text{mm}$ or $> 20\%$ frames exceeding $0.50\,\text{mm}$.
* **Site Partitioning:**
  * **Development LOIO Folds:** KKI, NeuroIMAGE, OHSU, Peking.
  * **Sealed External Test Institution:** **NYU** (scored strictly once after freezing all development models and baselines).
  * **A Priori Excluded Sites:** Pittsburgh (0% ADHD), WashU (0% ADHD), Brown (unreleased diagnostic labels in competition training set).

---

## 2. Official Download Source & Access Requirements

* **Primary Repository Platform:** Neuroimaging Informatics Tools and Resources Clearinghouse (NITRC).
* **Project Page:** The Neuro Bureau ADHD-200 Preprocessed Repository  
  * URL: [https://www.nitrc.org/projects/adhd200preproc/](https://www.nitrc.org/projects/adhd200preproc/)
* **File Release System (FRS):** Group ID 383  
  * URL: [https://www.nitrc.org/frs/?group_id=383](https://www.nitrc.org/frs/?group_id=383)
* **Access Credentials & Authentication:**
  * Accessing derivative packages on NITRC requires an authenticated NITRC user account (free registration).
  * Direct unauthenticated HTTP requests (e.g., standard `curl` or `wget`) receive an HTTP 302 redirect to `https://www.nitrc.org/account/login.php`.
  * **Download Procedure:** Download archives manually via an authenticated web browser session into a local staging directory.

---

## 3. Expected File Formats & Directory Structure

### 3.1. Distribution Packages
On NITRC, Athena derivatives are distributed as compressed tarballs:
1. **Consolidated Time-Series Archive:** `ADHD200_CC200_TCs_filtfix.tar.gz` (NITRC Release 1600 package title "CC200 Time Courses Corrected Filtering", also referenced historically without underscore as `ADHD200_CC200TCs_filtfix.tar.gz`; contains Craddock 200 regional time series across all sites).
2. **Per-Site Full Derivative Archives:** `Athena_<Site>.tar.gz` (e.g., `Athena_NYU.tar.gz`, `Athena_Peking_1.tar.gz`, `Athena_KKI.tar.gz`, `Athena_OHSU.tar.gz`, `Athena_NeuroIMAGE.tar.gz`).

### 3.2. Expected Directory Hierarchy
When unpacked into the local directory (`NeuroADHD/data/athena/`), files follow the standard Neuro Bureau structure:
```text
NeuroADHD/data/athena/
├── <Site_Name>/
│   └── <Subject_ID>/
│       └── session_1/
│           └── rest_1/
│               ├── <Subject_ID>_rest_1_rois_cc200.1D     <-- 200 ROI time-series
│               ├── <Subject_ID>_rest_1_motion.1D         <-- 6 motion parameters
│               └── snwmda_session_1_rest_1.nii.gz       <-- Voxelwise BOLD (not loaded)
```

### 3.3. File Specifications
* **Time-Series Files (`.1D`):** ASCII text table. Rows represent BOLD frames ($T \approx 120\text{--}260$ depending on site); whitespace-delimited columns represent mean BOLD signal per ROI ($K \approx 190\text{--}200$).
* **Motion Parameter Files (`.1D`):** ASCII text table output by AFNI `3dvolreg`. Exactly 6 whitespace-delimited columns:
  * Columns 1–3: Rotational displacements ($roll, pitch, yaw$) in degrees.
  * Columns 4–6: Translational displacements ($dS, dL, dP$) in millimeters.
* **Direct Availability:** Time-series and motion parameters are available **directly** as pre-extracted text files. **No raw 4D NIfTI preprocessing is required.**

---

## 4. Subject and Site Selection Rules

### 4.1. Comprehensive Cohort Inclusion Rule [REGISTERED PROTOCOL]
* Include **every subject with a valid diagnostic label (TDC or ADHD) and a usable `rest_1` scan**, pooling across both the competition training cohort and released-label test cohort if represented in the Athena release.
* Exclude participants with unreleased/missing diagnoses, corrupted time series, or zero valid BOLD frames.

### 4.2. Institutional Inclusion & Exclusion Criteria [REGISTERED PROTOCOL]
1. **Multi-Class Sites Included in Modeling ($N=5$ institutions):**
   * **NYU:** Candidate sealed validation site ($N \approx 216$ phenotypic records).
   * **Peking:** Development Fold 1 (pooled Peking 1, 2, 3; $N \approx 194$).
   * **KKI:** Development Fold 2 ($N \approx 83$).
   * **OHSU:** Development Fold 3 ($N \approx 79$; note: 14 controls seen in Study 1 exploration).
   * **NeuroIMAGE:** Development Fold 4 ($N \approx 48$).
2. **Single-Class / Unlabeled Sites Excluded a Priori ($N=3$ institutions):**
   * **Pittsburgh:** $N \approx 89$ (100% controls, 0% ADHD in training release; excluded).
   * **WashU:** $N \approx 59$ (100% controls, 0% ADHD in training release; excluded).
   * **Brown:** $N \approx 8$ (Diagnostic labels unreleased in competition training release; excluded).

---

## 5. Expected Sample Counts & Status

*All subject counts listed below are derived from ADHD-200 competition training phenotypic summary records and remain **[UNVERIFIED - Subject to file-level inspection]**:*

| Institution / Partition | Phenotypic TDC | Phenotypic ADHD | Phenotypic Total | Registered Role |
| :--- | :---: | :---: | :---: | :--- |
| **KKI** | 61 | 22 | 83 | Development LOIO Fold |
| **NeuroIMAGE** | 23 | 25 | 48 | Development LOIO Fold |
| **OHSU** | 42 | 37 | 79 | Development LOIO Fold |
| **Peking** (1, 2, 3) | 116 | 78 | 194 | Development LOIO Fold |
| **Development Subtotal** | **242** | **162** | **404** | **Pooled Development Cohort** |
| **NYU** | 98 | 118 | 216 | **Sealed External Validation Site** |
| **Total Analyzed Cohort** | **340** | **280** | **620** | *(Corrected total; not 640)* |

* **Mechanical Recomputation Requirement [REGISTERED PROTOCOL]:** Following physical archive download and inspection via `scripts/athena_inspect.py`, exact sample sizes and fold allocations will be recomputed mechanically from the written rule and committed to `docs/study2/COHORT_COUNTS_<DATE>.md`. This is a pre-registered procedure and is **not a protocol deviation**.

---

## 6. File Integrity Verification Strategy

Prior to feature extraction or model training, the following automated verification checks will be executed using `scripts/athena_inspect.py`:

1. **Archive Checksum Verification:** Compute SHA-256 checksums of downloaded tarball archives and record them in an archive manifest.
2. **Matrix Shape Validation:**
   * Verify each `.1D` file contains non-zero lines ($T \ge 50$ BOLD frames).
   * Verify column count corresponds to the CC200 parcellation ($K \ge 180$ columns).
   * Flag any non-numeric or `NaN` values.
3. **Motion Parameter Verification:**
   * Confirm motion file exists for each subject.
   * Verify that motion files contain exactly 6 columns.
   * Verify unit scale and orientation from file headers before computing Power's FD.
4. **Phenotype Matching Audit:**
   * Match subject identifiers against release phenotypic tables (`ScanDir ID`).
   * Verify that no diagnostic labels are missing or conflicting.
5. **Zero-ID Reporting Check:** Ensure all console outputs and logs report aggregate counts only, printing zero subject IDs.

---

## 7. Missing & Corrupted Data Handling Rules [REGISTERED PROTOCOL]

1. **Missing or Corrupted Scan Files:** Any subject lacking a complete `rest_1` time-series file or with corrupted/unreadable rows is dropped prior to cohort definition.
2. **Missing Motion Files:** Any subject lacking a corresponding 6-column motion file is dropped from the motion-audited analysis.
3. **Constant or Empty ROIs:**
   * In development: Any ROI exhibiting constant zero variance across even one development subject is dropped from the development ROI set.
   * In sealed NYU site: If an ROI in the development set is empty or constant in NYU, its correlation edges for that subject are set to zero ($r=0, z=0$). Development weights and feature selections are never modified.

---

## 8. NYU Sealed-Site Protection Protocol

To guarantee untainted external confirmation, the NYU site ($N \approx 216$) is protected by the following strict execution firewall:

* **Administrative Checks Only:** Inspection of NYU data is strictly limited to confirming file existence, file size, and motion pairing.
* **Zero Analytical Inspection:** No investigator or script may inspect NYU time-series values, visualize NYU correlation matrices, or compute exploratory statistics on NYU data.
* **Zero Model Tuning:** No model selection, hyperparameter tuning, feature selection, or imputation parameters may use NYU data.
* **Single Execution:** The frozen 4-step pipeline and Random Forest estimator are refitted once on the pooled development cohort and scored **exactly once** on NYU after the development report is completely finalized and committed.

---

## 9. Dataset Provenance, Ethics & Licensing

* **Dataset Initiative:** ADHD-200 Global Competition Consortium / International Neuroimaging Data-sharing Initiative (INDI).
* **Preprocessed Pipeline Author:** The Neuro Bureau (R. Cameron Craddock et al., 2011–2012).
* **Hosting Platform:** Neuroimaging Informatics Tools and Resources Clearinghouse (NITRC).
* **Licensing & Terms of Use:** Open-access for scientific research under the Creative Commons Attribution-NonCommercial (CC BY-NC) / Open Data Commons terms. Data is fully anonymized/de-identified compliant with HIPAA Safe Harbor regulations.
* **Research Use Only:** Derivatives are for neuroscience research purposes only; not clinical diagnostic tools.

---

## 10. Discrepancies Between Registered Protocol & Data Availability

| Dimension | Registered Protocol Requirement | Actual Data Availability (Verified) | Resolution / Pre-Registered Fallback |
| :--- | :--- | :--- | :--- |
| **Pipeline Origin** | Athena AFNI/FSL preprocessed derivatives | Available on NITRC (`group_id=383`) | Aligned. |
| **Atlas Parcellation** | Craddock 200 (CC200) | Available directly as `.1D` time series | Aligned. |
| **Preprocessing Variant** | Band-pass filtered, non-GSR (`filtfix`) | Available on NITRC (`ADHD200_CC200_TCs_filtfix.tar.gz`) | Aligned; fallback rule specified if variant missing. |
| **Motion Parameters** | 6-parameter motion files (`_motion.1D`) | Available in site packages | Aligned; column order verified prior to FD calculation. |
| **Download Access** | Direct download | Requires authenticated NITRC login | Staged for manual browser download to local directory. |
| **Sample Counts** | $N=404$ dev, $N=216$ NYU ($N=620$ total) | Unverified until physical file extraction | Mechanical recomputation rule formally registered in Section 4.2. |
