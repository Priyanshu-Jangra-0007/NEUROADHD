# [SUPERSEDED] Historical Audit Report

> [!WARNING]
> **SUPERSEDED / HISTORICAL AUDIT DOCUMENTATION**
> This report has been archived to `archive/audit_reports/` for historical reference and audit trail provenance.
> Key corrections to superseded claims in this document:
> 1. **Audit score superseded**: The preliminary rating of 81.8/100 is superseded by the follow-up site confound and generalization audit.
> 2. **"Evaluated strictly once" superseded**: The Peking_2 test set was scored 10 times in total across preliminary runs, locking iterations, and baseline comparisons.
> 3. **Probability baseline corrected**: The claimed 0.35–0.48 baseline probability range was incorrect; the empirical prediction probability range is 0.05–0.67.
> 4. **Batch holdout scope**: Peking_2 is a same-institution, same-scanner batch holdout (Peking_1 and Peking_3 are in the training split).
> 5. **Site confound collapse**: On the full 153-subject cohort under Leave-One-Institution-Out (LOIO) cross-validation, connectivity drops to near-chance (AUC 0.5342, p = 0.0859, not statistically significant).
> 6. **Exploratory subset**: The 118-subject clean subset (excluding zero-ADHD sites OHSU and Pittsburgh) was chosen after the full cohort showed no signal (exploratory). Demographic covariates (age and sex alone) achieve LOIO AUC 0.5976 [0.492, 0.702], while connectivity achieves 0.6221 [0.515, 0.724] (p = 0.021); the difference was not statistically tested.
> 
> Note: Superseded Study 1 audit note — the referenced standalone corrections file was not kept; see [`docs/study2/STUDY2_REPORT.md`](../../docs/study2/STUDY2_REPORT.md) and [`docs/PRESENTATION_OUTLINE.md`](../../docs/PRESENTATION_OUTLINE.md) for current corrected findings and defense notes.

---

# NEUROADHD: EXACT DATASET TRACE & FILE AUDIT REPORT

**Document ID:** `NEUROADHD-AUDIT-003`  
**Date of Audit:** 2026-11-15  
**Audit Standard:** Forensic Line-by-Line File and Data Flow Verification  
**Primary Focus:** Tracing the Complete Trajectory from Raw MRI Data to Final Model Training and Evaluation  

---

## EXECUTIVE SUMMARY & PROVENANCE MAP

This document provides a line-by-line, byte-verified audit tracing every input file, script, phenotypic join, feature extraction step, and partition manifest that constitutes the training and evaluation corpus of the **NeuroADHD** platform.

```
[Raw BIDS NIfTI Scans] ──► [Historical C-PAC Derivatives] ──► [roi_CC200.csv (161 Scans)]
                                                                    │
                                                                    ▼
[Phenotypic participants.tsv] ──► [cc200_subject_label_mapping.csv (154 Labeled)]
                                                                    │
                                                                    ▼
                                                    [QC: Exclude 0015038 (83 flat ROIs)]
                                                                    │
                                                                    ▼
                                             [extract_cc200_connectivity.py (153 Subjects)]
                                                                    │
                                                                    ▼
                                           [cc200_connectivity_features.csv (153 x 19,903)]
                                                                    │
                                    ┌───────────────────────────────┴───────────────────────────────┐
                                    ▼                                                               ▼
                    [subject_split_manifest.csv]                                    [cc200_site_holdout.json]
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
[Train Partition: 114 Subj]     [Validation Partition: 19 Subj]  [Locked Test Partition: 20 Subj]
- KKI, NYU, OHSU,               - NeuroIMAGE Site                - Peking_2 Site
  Peking_1/3, Pittsburgh        - 9 Control / 10 ADHD            - 10 Control / 10 ADHD
- 84 Control / 30 ADHD                      │                               │
    │                                       │                               │
    └───────────────────┬───────────────────┘                               │
                        ▼                                                   │
     [Refit Corpus for Final Model: 133 Subjects]                           │
                        │                                                   │
                        ▼                                                   ▼
            [train_model.py / run_final_test_evaluation.py] ──► [Peking_2 Locked Test]
```

---

## 1. ORIGINAL RAW MRI FILES AUDIT

### 1.1 Source Repository & Storage Locations
- **Data Repository:** ADHD-200 Consortium Public S3 Bucket (`fcp-indi/data/Projects/ADHD200/RawDataBIDS/`).
- **Local Storage Locations:**
  1. `<repo>/outputs/ADHD200_RawDataBIDS/`:
     - Contains BIDS-compliant sub-folders for two sites: `Brown/` (24 subjects) and `KKI/` (37 subjects).
     - Verified by `outputs/ADHD200_RawDataBIDS_manifest.tsv` ($275,545\text{ bytes}$) and `outputs/ADHD200_RawDataBIDS_file_list.txt` ($193,232\text{ bytes}$).
  2. `<repo>/ADHD200_Official/experiments/raw_pilot_18/raw/BIDS/`:
     - Contains 18 selected subjects across 9 international sites (2 subjects per site).
     - Exactly 85 files totaling $1,151,341,169\text{ bytes}$ ($1.07\text{ GiB}$).

### 1.2 File Format & BIDS Specifications
- **Structural Scans:** 3D T1-weighted magnetization-prepared rapid gradient-echo (MPRAGE) or equivalent structural scans in compressed NIfTI-1 format (`sub-<id>_T1w.nii.gz`).
- **Functional Scans:** 4D BOLD resting-state scans (`sub-<id>_task-rest_bold.nii.gz` or `sub-<id>_task-rest_run-1_bold.nii.gz`).
- **BIDS Sidecars:** Accompanying JSON sidecars (`task-rest_bold.json`, `T1w.json`) detailing scanner metadata, including:
  - `RepetitionTime` ($TR$): Ranging from $1.5\text{ s}$ to $2.5\text{ s}$ depending on site.
  - `TaskName`: `"rest"`.
  - `FlipAngle`, `EchoTime`, `SliceTiming`.

---

## 2. ROI TIME-SERIES INPUT AUDIT

### 2.1 Physical Storage of CC200 Derivatives
- **Local Path:** `<repo>/NeuroADHD/data/adhd200/raw/cc200/`
- **Total Sessions Downloaded:** Exactly **161 subject session directories** (e.g., `0010020_session_1/`, `1017176_session_1/`).
- **Per-Subject File:** Each directory contains exactly one `roi_CC200.csv` file.

### 2.2 Internal CSV Structure & Dimensionality
Inspected via `NeuroADHD/src/features/extract_cc200_connectivity.py` (`load_roi_timeseries`, lines 26–48):
- **Orientation:** Rows are ROIs; columns are temporal acquisition frames (TRs).
- **Column 0:** ROI index (`node/volume`), containing integers $1, \dots, 200$ strictly in ascending order.
- **Columns 1 to $T$:** Continuous floating-point BOLD intensity values for temporal frames $t \in \{1, \dots, T\}$.
- **Temporal Dimensions ($T$):** Varies from $77$ frames to $261$ frames across 13 distinct scan durations.
- **Missing Data:** Zero empty cells, zero non-numeric tokens.

### 2.3 Quality Control & Subject Exclusion
- **Excluded Subject:** Subject **`0015038`** (Session: `0015038_session_1`, Site: `WashU`).
- **Exclusion Mechanism:** `extract_cc200_connectivity.py` lines 88–92 computes temporal standard deviation:
  ```python
  std = np.std(time_by_roi, axis=0, ddof=0)
  constant_count = int(np.count_nonzero(std == 0.0))
  ```
- **Finding:** Subject `0015038` contained **83 constant (zero-variance) ROI time courses**, rendering Pearson correlation mathematically undefined ($\frac{0}{0}$).
- **Audit Record:** Automatically logged to `NeuroADHD/data/adhd200/features/cc200/cc200_skipped_subjects.csv`:
  ```csv
  subject_id,reason,input
  0015038,83 ROI time series are constant; Pearson correlations undefined,/Users/.../data/adhd200/raw/cc200/0015038_session_1/roi_CC200.csv
  ```

---

## 3. PHENOTYPE & DIAGNOSIS MAPPING AUDIT

### 3.1 Source Phenotypic Files
Located under `<repo>/NeuroADHD/data/adhd200/raw/phenotypes/`:
- `Brown/participants.tsv`, `Brown_TestRelease_phenotypic.csv`
- `KKI/participants.tsv`, `KKI_phenotypic.csv`
- `NeuroIMAGE/participants.tsv`, `NeuroIMAGE_phenotypic.csv`
- `NYU/participants.tsv`, `NYU_phenotypic.csv`
- `OHSU/participants.tsv`, `OHSU_phenotypic.csv`, `OHSU_TestRelease_phenotypic.csv`
- `Peking_1/participants.tsv`, `Peking_1_phenotypic.csv`, `Peking_1_TestRelease_phenotypic.csv`
- `Peking_2/participants.tsv`
- `Peking_3/participants.tsv`
- `Pittsburgh/participants.tsv`, `Pittsburgh_phenotypic.csv`
- `WashU/participants.tsv`

### 3.2 Subject-to-Diagnosis Mapping Master File
- **Path:** `<repo>/NeuroADHD/data/adhd200/interim/cc200_subject_label_mapping.csv`
- **Total Audited Subjects:** 161 entries.
- **Status Counts:**
  - `matched_labeled`: **154 subjects**
  - `missing_or_pending_diagnosis`: **7 subjects** (All 7 are Brown University subjects whose diagnostic status was pending/unreleased in the ADHD-200 release: subjects `0026001`, `0026002`, `0026004`, `0026005`, `0026009`, `0026014`, `0026015`).
- **Binary Diagnostic Encoding (DSM-IV / DSM-5):**
  - Raw Value `0` (Typically Developing Control) $\longrightarrow$ **Control (`0`)** ($N = 104$)
  - Raw Value `1` (ADHD-Combined) $\longrightarrow$ **ADHD (`1`)**
  - Raw Value `2` (ADHD-Hyperactive/Impulsive) $\longrightarrow$ **ADHD (`1`)**
  - Raw Value `3` (ADHD-Inattentive) $\longrightarrow$ **ADHD (`1`)**
  - Total ADHD cases among 154 usable subjects: **50 subjects**.

---

## 4. FEATURE EXTRACTION PIPELINE AUDIT

### 4.1 Script Provenance
- **Script Path:** `<repo>/NeuroADHD/src/features/extract_cc200_connectivity.py`
- **Execution Log:** Executed on 2026-09-29T19:33:49 UTC; summary stored in `cc200_extraction_summary.json`.

### 4.2 Exact Extraction Algorithm
1. **Transposition (Line 45):**
   The raw matrix is $(200 \text{ ROIs} \times T \text{ timepoints})$. It is transposed to $(T \times 200)$ so columns represent random variables (ROIs).
2. **Pearson Correlation Matrix (Line 94):**
   ```python
   corr = np.corrcoef(time_by_roi, rowvar=False)  # Shape: (200, 200)
   corr = (corr + corr.T) / 2.0                   # Symmetrize numerical noise
   np.fill_diagonal(corr, 1.0)                    # Force exact diagonal 1.0
   ```
3. **Upper Triangle Edge Extraction (Lines 65, 102):**
   ```python
   tri_i, tri_j = np.triu_indices(200, k=1)
   edge_corr = corr[tri_i, tri_j]                 # Shape: (19900,)
   ```
4. **Fisher $r$-to-$z$ Transformation with Epsilon Guard (Lines 22, 103–105):**
   ```python
   FISHER_EPSILON = 1e-7
   clipped = np.clip(edge_corr, -1.0 + FISHER_EPSILON, 1.0 - FISHER_EPSILON)
   z = np.arctanh(clipped)
   ```
5. **Feature Ordering & Naming (Lines 63–64):**
   Ordered row-major lexicographically:
   `ROI_001__ROI_002`, `ROI_001__ROI_003`, ..., `ROI_001__ROI_200`,
   `ROI_002__ROI_003`, ..., `ROI_199__ROI_200`.
   Exact feature mapping verified in `NeuroADHD/data/adhd200/features/cc200/cc200_feature_metadata.csv` ($19,900$ rows).

---

## 5. FINAL TRAINING MATRIX AUDIT

### 5.1 Authoritative Matrix File
- **Full Path:** `<repo>/NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv`
- **File Size on Disk:** $47,605,745\text{ bytes}$ ($\approx 47.6\text{ MB}$)
- **Row Count:** Exactly **153 subjects** (154 labeled minus 1 zero-variance QC exclusion).
- **Column Count:** Exactly **19,903 columns**.

### 5.2 Column Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 cc200_connectivity_features.csv SCHEMA                      │
├─────────────────────┬──────────────┬──────────────┬─────────────────────────┤
│ Column Index        │ Column Name  │ Data Type    │ Role                    │
├─────────────────────┼──────────────┼──────────────┼─────────────────────────┤
│ Column 0            │ subject_id   │ string (id)  │ Subject ID key (Metadata│
│ Column 1            │ site         │ string       │ Acquisition site (Meta) │
│ Columns 2 to 19,901 │ ROI_001__... │ float64      │ 19,900 Fisher-z FC Edges│
│ Column 19,902       │ diagnosis    │ int64        │ Binary Target (0 or 1)  │
└─────────────────────┴──────────────┴──────────────┴─────────────────────────┘
```

- **Target Class Distribution:**
  - Control (`0`): **103 subjects** ($67.3\%$)
  - ADHD (`1`): **50 subjects** ($32.7\%$)
  - Total: **153 subjects**

---

## 6. TRAIN / VALIDATION / TEST MANIFESTS AUDIT

### 6.1 Split Strategy & Site Partitioning
Defined in `<repo>/NeuroADHD/configs/splits/cc200_site_holdout.json` and generated by `NeuroADHD/src/preprocessing/create_cc200_splits.py`.

The strategy is a **fixed whole-site holdout**. To test genuine out-of-site cross-scanner generalizability, entire scanning centers are assigned to individual partitions.

### 6.2 Partition Breakdown & Demographics

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                     PARTITION COMPOSITION & SITE AUDIT                        │
├────────────┬──────────┬─────────┬───────┬─────────────────────────────────────┤
│ Partition  │ Subjects │ Control │ ADHD  │ Sites Represented (Counts)          │
├────────────┼──────────┼─────────┼───────┼─────────────────────────────────────┤
│ Train      │ 114      │ 84      │ 30    │ KKI (24), NYU (6), OHSU (14),       │
│            │          │         │       │ Peking_1 (38), Peking_3 (11),       │
│            │          │         │       │ Pittsburgh (21)                     │
│ Validation │ 19       │ 9       │ 10    │ NeuroIMAGE (19)                     │
│ Test       │ 20       │ 10      │ 10    │ Peking_2 (20)                       │
├────────────┼──────────┼─────────┼───────┼─────────────────────────────────────┤
│ TOTAL      │ 153      │ 103     │ 50    │ 8 Unique International Sites        │
└────────────┴──────────┴─────────┴───────┴─────────────────────────────────────┘
```

### 6.3 Site-Disjoint Verification
As verified in `NeuroADHD/data/adhd200/interim/cc200_split_validation_report.json`:
- Subject Overlap:
  - $\text{Train} \cap \text{Validation} = \emptyset$ (0 subjects)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 subjects)
  - $\text{Validation} \cap \text{Test} = \emptyset$ (0 subjects)
- Site Overlap:
  - $\text{Sites}_{\text{Train}} \cap \text{Sites}_{\text{Validation}} = \emptyset$
  - $\text{Sites}_{\text{Train}} \cap \text{Sites}_{\text{Test}} = \emptyset$
  - $\text{Sites}_{\text{Validation}} \cap \text{Sites}_{\text{Test}} = \emptyset$
- **Conclusion:** Partitions are 100% subject-disjoint and 100% site-disjoint.

---

## 7. EXPLICIT ANSWERS TO THE 6 CORE AUDIT QUESTIONS

### Question 1: WHAT EXACT CSV FILE IS MY MODEL TRAINING ON?
**Answer:**  
The model trains on the precomputed feature table:  
`<repo>/NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv`  
Specifically:
- During baseline and optimization benchmarks, classifiers train on the **114 subject rows** specified in `<repo>/NeuroADHD/data/adhd200/processed/splits/train_subjects.csv`.
- For the authoritative final locked model (`final_model.joblib`), the model was refit on the **133 combined Train + Validation subjects** ($114 + 19 = 133$) filtered from this same CSV.

### Question 2: WHAT EXACT SCRIPT LOADS THAT CSV?
**Answer:**
- In the benchmark suite:  
  `NeuroADHD_ModelBenchmark/scripts/train_model.py` (lines 81–82) and `NeuroADHD_ModelBenchmark/scripts/run_all_models.py` (lines 78–79).
- In the final locked evaluation:  
  `NeuroADHD_ModelBenchmark/scripts/run_final_test_evaluation.py` (lines 86–87).
- In the core preprocessing module:  
  `NeuroADHD/src/preprocessing/cc200_ml_preprocessing.py` (function `load_cc200_xy`, line 44).

### Question 3: WHAT EXACT FUNCTION SEPARATES X AND Y?
**Answer:**  
In `NeuroADHD/src/preprocessing/cc200_ml_preprocessing.py`, the function `load_cc200_xy(feature_table)` (lines 42–71) executes the exact separation:
```python
feature_columns = [column for column in table.columns if column.startswith("ROI_")]
X = table.loc[:, feature_columns].astype("float64")
y = table["diagnosis"].astype("int64")
metadata = table.loc[:, ["subject_id", "site"]].copy()
```
In `NeuroADHD_ModelBenchmark/scripts/train_model.py` (lines 88–98):
```python
X_full = table.loc[:, feature_cols].astype("float64")
y_full = table["diagnosis"].astype("int64")
X_train = X_full.loc[train_ids]
y_train = y_full.loc[train_ids].to_numpy()
```

### Question 4: WHAT EXACT SCRIPT CALLS model.fit()?
**Answer:**
- For baseline models and hyperparameter trials:  
  `NeuroADHD_ModelBenchmark/scripts/train_model.py` at **Line 140**:
  ```python
  clf.fit(X_train_prep, y_train)
  ```
- For the final locked model:  
  `NeuroADHD_ModelBenchmark/scripts/run_final_test_evaluation.py` at **Line 132**:
  ```python
  final_model.fit(X_train_val_trans, y_train_val)
  ```
- For voting and stacking ensembles:  
  `NeuroADHD_ModelBenchmark/scripts/run_all_models.py` at lines **119**, **137**, and **158**:
  ```python
  soft_vote.fit(X_train_prep, y_train)
  hard_vote.fit(X_train_prep, y_train)
  stacking.fit(X_train_prep, y_train)
  ```

### Question 5: WHAT EXACT FILES ARE USED FOR VALIDATION?
**Answer:**
- Subject manifest:  
  `<repo>/NeuroADHD/data/adhd200/processed/splits/validation_subjects.csv` (19 subjects from the `NeuroIMAGE` site: 9 Controls, 10 ADHD).
- Feature vectors:  
  The corresponding 19 rows extracted from `cc200_connectivity_features.csv`.
- Transformed by:  
  `preproc.transform(X_val)` in `train_model.py` (line 111).
- Validation prediction records:  
  `NeuroADHD_ModelBenchmark/runs/<run_id>/predictions/<model>_validation_predictions.csv`.

### Question 6: WHAT EXACT FILES ARE USED FOR FINAL TESTING?
**Answer:**
- Subject manifest:  
  `<repo>/NeuroADHD/data/adhd200/processed/splits/test_subjects.csv` (20 subjects from the `Peking_2` site: 10 Controls, 10 ADHD).
- Feature vectors:  
  The corresponding 20 rows extracted from `cc200_connectivity_features.csv`.
- Final evaluation execution script:  
  `NeuroADHD_ModelBenchmark/scripts/run_final_test_evaluation.py`.
- Ground truth test outputs:  
  `<repo>/NeuroADHD_ModelBenchmark/results/final_locked/final_test_results.json` and `final_test_predictions.csv`.
