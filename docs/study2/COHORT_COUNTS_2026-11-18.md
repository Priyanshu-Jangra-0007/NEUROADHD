# Study 2 Mechanical Cohort Freeze & Motion Attrition Report (2026-11-18)

**Date:** 2026-11-18  
**Branch:** `feat/study2-athena-acquisition`  
**Preregistration Reference:** OSF Preregistration ([https://osf.io/grbt2](https://osf.io/grbt2), DOI: `10.17605/OSF.IO/GRBT2`)  
**Locked Protocol Hash (SHA-256, unmodified):** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`  
**Protocol Deviation Applied:** [`docs/study2/DEVIATIONS.md`](DEVIATIONS.md) Branch (b) — NIAK-derived Power et al. (2012) FD (volume-centered, first 3 frames dropped, $r = 50\text{ mm}$, $\text{FD}_0 = 0.0$), with no-exclusion elevated to co-primary sensitivity.  
**Script:** [`scripts/generate_cohort_counts.py`](../../scripts/generate_cohort_counts.py)  
**Guardrails:** Diagnosis labels used strictly for aggregate counts and attrition tests only; zero modeling; zero individual subject IDs logged or printed; **NYU archive counts derived strictly from CC200 tar member names only (zero NYU imaging opened or extracted) joined to the phenotype TSV**.

---

## 1. Verbatim Cohort & Sealing Rules from `PREREGISTRATION.md`

### Cohort Inclusion & Exclusion Rule (`PREREGISTRATION.md`, Section 4.1)
> * **Rule:** Include **every subject with a valid diagnostic label (TDC or ADHD) and a usable `rest_1` scan**, pooling across both the competition training cohort and the released-label test cohort if the Athena release contains them.
> * **Excluded Categories:** Subjects with unreleased/missing diagnoses, uninterpretable scans, or scans failing file integrity checks. Single-class institutions (e.g., WashU and Pittsburgh, containing 100% controls and 0% ADHD in the competition release) are excluded a priori from classification folds.

### Mechanical Recomputation Rule (`PREREGISTRATION.md`, Section 4.2)
> * **Mechanical Recomputation:** After physical archive inspection via `scripts/athena_inspect.py`, exact sample counts, the designated sealed institution, and the development cross-validation folds will be **recomputed mechanically strictly from the pre-registered written sealing rule**.

### Written Sealing Rule (`PREREGISTRATION.md`, Section 4.3)
> *The sealed institution must have balanced representation of both diagnostic classes (>5% ADHD), standard pediatric acquisition parameters, and minimal or zero exposure during historical pipeline exploration.*

### Motion Exclusion Rules (`PREREGISTRATION.md`, Sections 5.3–5.4)
> * **Primary Exclusion Threshold:** A scan is excluded if:
>   $$\text{Mean FD} > 0.50\,\text{mm} \quad \text{OR} \quad \text{Percentage of frames with } \text{FD} > 0.50\,\text{mm} > 20\%$$
> * **Stricter Motion Threshold (Sensitivity):** $\text{Mean FD} > 0.20\,\text{mm}$ or $> 10\%$ of frames with $\text{FD} > 0.20\,\text{mm}$ excluded.
> * **No Motion Exclusion (Co-Primary Sensitivity under `DEVIATIONS.md` Branch b):** All participants with complete time series retained regardless of motion parameters.

---

## 2. All Verified Development Scans & Sealed NYU Count (Member Names Only)

All **404 / 404** development `rest_1` scans in `ADHD200_CC200_TCs_filtfix.tar.gz` matched a valid binary diagnosis in `adhd200_preprocessed_phenotypics.tsv` (**0 missing or pending diagnoses**; `DX = 0` mapped to TDC, `DX ∈ {1, 2, 3}` mapped to ADHD).

| Site / Institution | Role | Total ($N$) | TDC ($n$) | ADHD ($n$) | % ADHD | Class $< 15$ Flag |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **KKI** | LOIO Fold 1 (Dev) | 83 | 61 | 22 | 26.5% | OK |
| **NeuroIMAGE** | LOIO Fold 2 (Dev) | 48 | 23 | 25 | 52.1% | OK |
| **OHSU** | LOIO Fold 3 (Dev) | 79 | 42 | 37 | 46.8% | OK |
| **Peking_1** | *(Peking sub-site)* | 85 | 61 | 24 | 28.2% | OK |
| **Peking_2** | *(Peking sub-site)* | 67 | 32 | 35 | 52.2% | OK |
| **Peking_3** | *(Peking sub-site)* | 42 | 23 | 19 | 45.2% | OK |
| **Peking (Combined)** | LOIO Fold 4 (Dev) | 194 | 116 | 78 | 40.2% | OK |
| **POOLED DEVELOPMENT** | **4-Institution LOIO** | **404** | **242** | **162** | **40.1%** | **OK** |
| **NYU *(CC200 archive member names only)*** | **Sealed External Site (Actual Scans)** | **216** | **98** | **118** | **54.6%** | **OK** |
| *(NYU phenotype TSV total — NOT scored $N$)* | *(Training + test rows in TSV)* | *(263)* | *(111)* | *(152)* | *(57.8%)* | — |

*(Note on NYU: Joining the 216 NYU `sfnwmrda*_session_1_rest_1_cc200_TCs.1D` member names in `ADHD200_CC200_TCs_filtfix.tar.gz` — read via `tarfile.getmembers()` only, with zero NYU imaging files opened or extracted — to `adhd200_preprocessed_phenotypics.tsv` yields exactly **216 scans (98 TDC, 118 ADHD, 54.6% ADHD; 0 missing DX)**. The 263 NYU rows in the consolidated phenotype TSV include 47 additional test-release rows absent from `ADHD200_CC200_TCs_filtfix.tar.gz`, so 263 is the phenotype table count, **not** the scored $N$. Single-class/near-single-class sites WashU [61 TDC, 0 ADHD] and Pittsburgh [94 TDC, 4 ADHD = 4.1% ADHD in TSV; 0 ADHD in training archive] and pending-label Brown [26 pending] are excluded a priori.)*

---

## 3. Site $\times$ Diagnosis Counts Across Motion Regimes & Fisher Exact Attrition Tests

### 3.1. Co-Primary No-Exclusion Sensitivity (`DEVIATIONS.md` Branch b)
Identical to Section 2 above: **404 / 404** development scans retained (242 TDC, 162 ADHD; 0 excluded; **0 folds with $< 15$ subjects in a class**).

### 3.2. Primary Motion Exclusion ($\text{Mean FD} > 0.50\text{ mm}$ OR $>20\%$ frames $> 0.50\text{ mm}$)

| Site / Institution | Retained Total | Retained TDC | Retained ADHD | Excluded Total | Excluded TDC ($n$ / %) | Excluded ADHD ($n$ / %) | Fisher Exact $p$ (2-sided) | Fold $< 15$ Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KKI** | 79 | 60 | 19 | 4 | 1 (1.6%) | 3 (13.6%) | `0.0551` | OK |
| **NeuroIMAGE** | 42 | 23 | 19 | 6 | 0 (0.0%) | 6 (24.0%) | **`0.0227`** | OK |
| **OHSU** | 58 | 33 | 25 | 21 | 9 (21.4%) | 12 (32.4%) | `0.3140` | OK |
| **Peking_1** | 84 | 61 | 23 | 1 | 0 (0.0%) | 1 (4.2%) | `0.2824` | OK |
| **Peking_2** | 64 | 30 | 34 | 3 | 2 (6.2%) | 1 (2.9%) | `0.6025` | OK |
| **Peking_3** | 41 | 23 | 18 | 1 | 0 (0.0%) | 1 (5.3%) | `0.4524` | OK |
| **Peking (Combined)** | 189 | 114 | 75 | 5 | 2 (1.7%) | 3 (3.8%) | `0.3930` | OK |
| **POOLED DEV** | **368** | **230** | **138** | **36** | **12 (5.0%)** | **24 (14.8%)** | **`0.0011`** | **OK** |

### 3.3. Strict Motion Exclusion ($\text{Mean FD} > 0.20\text{ mm}$ OR $>10\%$ frames $> 0.20\text{ mm}$)

| Site / Institution | Retained Total | Retained TDC | Retained ADHD | Excluded Total | Excluded TDC ($n$ / %) | Excluded ADHD ($n$ / %) | Fisher Exact $p$ (2-sided) | Fold $< 15$ Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KKI** | 21 | 19 | **2** | 62 | 42 (68.9%) | 20 (90.9%) | **`0.0482`** | **FLAG (`ADHD = 2 < 15`)** |
| **NeuroIMAGE** | 19 | **11** | **8** | 29 | 12 (52.2%) | 17 (68.0%) | `0.3767` | **FLAG (`TDC = 11, ADHD = 8 < 15`)** |
| **OHSU** | 19 | **13** | **6** | 60 | 29 (69.0%) | 31 (83.8%) | `0.1872` | **FLAG (`TDC = 13, ADHD = 6 < 15`)** |
| **Peking_1** | 38 | 35 | **3** | 47 | 26 (42.6%) | 21 (87.5%) | **`0.0002`** | **FLAG (`ADHD = 3 < 15`)** |
| **Peking_2** | 28 | 15 | **13** | 39 | 17 (53.1%) | 22 (62.9%) | `0.4648` | **FLAG (`ADHD = 13 < 15`)** |
| **Peking_3** | 8 | **5** | **3** | 34 | 18 (78.3%) | 16 (84.2%) | `0.7092` | **FLAG (`TDC = 5, ADHD = 3 < 15`)** |
| **Peking (Combined)** | 74 | 55 | 19 | 120 | 61 (52.6%) | 59 (75.6%) | **`0.0015`** | OK *(pooled fold)* |
| **POOLED DEV** | **133** | **98** | **35** | **271** | **144 (59.5%)** | **127 (78.4%)** | **`0.0001`** | **OK *(pooled only)*** |

---

## 4. Key Attrition, Fold-Size & Sealing-Rule Findings

1. **Differential Motion Attrition by Diagnosis:**
   - Under the **primary threshold**, ADHD participants are excluded at nearly 3$\times$ the rate of TDC participants pooled (**14.8% vs. 5.0%**, two-sided Fisher exact **$p = 0.0011$**), driven significantly by **NeuroIMAGE** (24.0% vs. 0.0%, $p = 0.0227$) and borderline by KKI (13.6% vs. 1.6%, $p = 0.0551$).
   - Under the **strict threshold**, pooled attrition reaches **78.4% ADHD vs. 59.5% TDC** (**$p = 0.0001$**), with severe differential loss in Peking_1 ($p = 0.0002$), Peking combined ($p = 0.0015$), and KKI ($p = 0.0482$).
   - This statistically significant differential attrition further underscores the scientific necessity of reporting the **co-primary no-exclusion sensitivity analysis** (`DEVIATIONS.md` Branch b) alongside primary exclusion.

2. **Folds Flagged for $< 15$ Subjects in a Class:**
   - **No Exclusion ($N = 404$):** **None** (minimum class size across LOIO folds = 22).
   - **Primary Exclusion ($N = 368$):** **None** (minimum class size across LOIO folds = 19 in KKI ADHD and NeuroIMAGE ADHD; minimum sub-site class = 18 in Peking_3 ADHD).
   - **Strict Exclusion ($N = 133$):** **3 of 4 LOIO institution folds fail the $\ge 15$ class-size floor**:
     - `KKI` (ADHD = **2**)
     - `NeuroIMAGE` (TDC = **11**, ADHD = **8**)
     - `OHSU` (TDC = **13**, ADHD = **6**)
     - *(And all three Peking sub-sites individually have $< 15$ in at least one class, though pooled Peking retains 55 TDC / 19 ADHD.)*
     - Strict-threshold LOIO is therefore **descriptive/attrition-only** and is **not run as a LOIO classification model** (recorded in [`DEVIATIONS.md`](DEVIATIONS.md)).

3. **Mechanical Outcome of the Written Sealing Rule:**
   - Applying `PREREGISTRATION.md` Section 4.3 strictly to these counts **confirms NYU as the sealed external validation institution**:
     - Balanced diagnostic classes ($>5\%$ ADHD): **PASS** (216 actual `rest_1` scans in CC200 archive: **98 TDC, 118 ADHD = 54.6% ADHD**; 263 total rows in phenotype TSV: 111 TDC, 152 ADHD).
     - Standard pediatric acquisition parameters: **PASS**.
     - Minimal exposure during Study 1 exploration: **PASS** (6 NYU subjects vs. 126 development subjects across KKI [24], NeuroIMAGE [19], Peking [69], OHSU [14]).
   - The four-institution LOIO development cohort is frozen as **KKI, NeuroIMAGE, OHSU, and Peking (1–3)**.
