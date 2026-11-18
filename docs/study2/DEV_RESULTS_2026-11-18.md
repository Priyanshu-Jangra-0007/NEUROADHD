# Study 2 Development Cohort LOIO Results (2026-11-18)

**Date:** 2026-11-18  
**Branch:** `feat/study2-athena-acquisition`  
**Preregistration Reference:** OSF Preregistration ([https://osf.io/grbt2](https://osf.io/grbt2), DOI: `10.17605/OSF.IO/GRBT2`)  
**Locked Protocol SHA-256 (unmodified):** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`  
**Frozen Analysis Code Commit:** `6e812b7ddbd34321ad767c645fe5db3c7235a7ef` (`scripts/run_study2_dev_analysis.py`, SHA-256: `fa4b41d2966e8219e93ac3a6151b1c141d10ef34af167771607edb88820d50b1`)  
**Analysis Freeze Commit / Tag:** `3474758ca946785eb575c2c07eb2b81127d67e32` (`study2-dev-analysis-frozen`)  
**Pre-Run Deviation Commit:** `1bcbff3` ([`docs/study2/DEVIATIONS.md`](DEVIATIONS.md), Branch b NIAK volume-centered FD with first 3 frames removed)  
**Sealed Site Status:** NYU untouched; zero NYU imaging, motion, or features opened.

> Results reported as produced by the frozen script; no analysis choices were changed after seeing them.

---

## 1. Execution Artifacts & Checksums (Git-Ignored under `NeuroADHD/data/athena/results/`)

- **Command Executed (Once, Locked Defaults):** `PYTHONPATH=. ./NeuroADHD/.venv/bin/python scripts/run_study2_dev_analysis.py --execute-real-dev`
- **Exit Code:** `0` (zero errors, zero retries)
- **Results JSON (`NeuroADHD/data/athena/results/study2_dev_results.json`, 4,225 bytes) SHA-256:**  
  `6d072448bf6573cd7f6e0bd65c0b6a141bc2cf60f2ea0e4a69bcc504918dc0d5`
- **Run Log (`NeuroADHD/data/athena/results/study2_dev_run.log`, 4,225 bytes) SHA-256:**  
  `6d072448bf6573cd7f6e0bd65c0b6a141bc2cf60f2ea0e4a69bcc504918dc0d5`
- **File Distinction Note (Verified 2026-11-18):** `study2_dev_results.json` and `study2_dev_run.log` are two separate files (distinct inodes); their hashes match because `scripts/run_study2_dev_analysis.py` wrote the JSON text plus newline to the JSON file and printed the same JSON text to stdout with zero stderr output.

---

## 2. Primary Analysis — `primary_368` ($N = 368$, Registered Section 5.3 Motion Exclusion)

Cohort: 368 scans (`KKI`: 79 [60 TDC / 19 ADHD], `NeuroIMAGE`: 42 [23 / 19], `OHSU`: 58 [33 / 25], `Peking`: 189 [114 / 75]; 17,955 Fisher-$z$ edges, $k = 100$ selected inside each fold).

### 2.1. Pooled LOIO ROC-AUC, Baselines & Paired Contrasts ($B_{\text{boot}} = 2,000$, $B_{\text{perm}} = 1,000$)

| Metric / Endpoint | Point Estimate | 95% Stratified Bootstrap CI | Permutation $p$ / Status |
| :--- | :---: | :---: | :---: |
| **H1: fMRI Pooled LOIO ROC-AUC** | **0.5241** (`0.524134`) | **`[0.4691, 0.5797]`** | **`p_perm = 0.2298`** (`229 / 1000` exceed; null mean `0.4924`, SD `0.0399`) |
| **Motion-Only Baseline Pooled LOIO AUC** | **0.6038** (`0.603812`) | `[0.5508, 0.6585]` | — |
| **Age & Sex Baseline Pooled LOIO AUC** | **0.6163** (`0.616273`) | `[0.5650, 0.6672]` | — |
| **H2: $\Delta\text{AUC}_{\text{motion}}$ ($\text{fMRI} - \text{Motion}$)** | **-0.0797** (`-0.079679`) | **`[-0.1520, -0.0099]`** | $\text{CI}_{\text{lo}} > 0$: **False** |
| **H3: $\Delta\text{AUC}_{\text{age/sex}}$ ($\text{fMRI} - \text{Age/Sex}$)** | **-0.0921** (`-0.092139`) | **`[-0.1649, -0.0169]`** | $\text{CI}_{\text{lo}} > 0$: **False** |

### 2.2. Per-Institution fMRI LOIO ROC-AUCs ($N = 368$)

| Institution | fMRI LOIO AUC | 95% Stratified Bootstrap CI | Motion Baseline AUC | Age/Sex Baseline AUC |
| :--- | :---: | :---: | :---: | :---: |
| **KKI** ($n = 79$) | **0.5500** (`0.550000`) | `[0.3930, 0.7026]` | `0.5825` | `0.4939` |
| **NeuroIMAGE** ($n = 42$) | **0.5698** (`0.569794`) | `[0.3959, 0.7529]` | `0.6018` | `0.6934` |
| **OHSU** ($n = 58$) | **0.6048** (`0.604848`) | `[0.4521, 0.7430]` | `0.6315` | `0.6279` |
| **Peking** ($n = 189$) | **0.5346** (`0.534620`) | `[0.4537, 0.6152]` | `0.6230` | `0.6730` |

### 2.3. Section 8.1 Development Success Criteria (`primary_368`)
- **Criterion 1** ($p_{\text{perm}} < 0.05$): **NOT MET** (`0.2298 >= 0.05`).
- **Criterion 2** ($\text{CI}_{95\%,\text{lo}}[\Delta\text{AUC}_{\text{motion}}] > 0$): **NOT MET** (`-0.1520 <= 0`).
- **Combined Section 8.1 Criterion:** **NOT MET** (`meets_dev_success_criteria = False`).

---

## 3. Registered Sensitivity Analysis — `no_exclusion_404` ($N = 404$, Registered Section 5.4 No Exclusion)

Cohort: 404 scans (`KKI`: 83 [61 TDC / 22 ADHD], `NeuroIMAGE`: 48 [23 / 25], `OHSU`: 79 [42 / 37], `Peking`: 194 [116 / 78]; 17,955 Fisher-$z$ edges, $k = 100$ selected inside each fold).

### 3.1. Pooled LOIO ROC-AUC, Baselines & Paired Contrasts ($B_{\text{boot}} = 2,000$, $B_{\text{perm}} = 1,000$)

| Metric / Endpoint | Point Estimate | 95% Stratified Bootstrap CI | Permutation $p$ / Status |
| :--- | :---: | :---: | :---: |
| **H1: fMRI Pooled LOIO ROC-AUC** | **0.5740** (`0.574023`) | **`[0.5243, 0.6243]`** | **`p_perm = 0.0050`** (`4 / 1000` exceed; null mean `0.4830`, SD `0.0359`) |
| **Motion-Only Baseline Pooled LOIO AUC** | **0.6158** (`0.615805`) | `[0.5670, 0.6644]` | — |
| **Age & Sex Baseline Pooled LOIO AUC** | **0.5857** (`0.585667`) | `[0.5369, 0.6363]` | — |
| **H2: $\Delta\text{AUC}_{\text{motion}}$ ($\text{fMRI} - \text{Motion}$)** | **-0.0418** (`-0.041781`) | **`[-0.0890, +0.0087]`** | $\text{CI}_{\text{lo}} > 0$: **False** |
| **H3: $\Delta\text{AUC}_{\text{age/sex}}$ ($\text{fMRI} - \text{Age/Sex}$)** | **-0.0116** (`-0.011644`) | **`[-0.0795, +0.0568]`** | $\text{CI}_{\text{lo}} > 0$: **False** |

### 3.2. Per-Institution fMRI LOIO ROC-AUCs ($N = 404$)

| Institution | fMRI LOIO AUC | 95% Stratified Bootstrap CI | Motion Baseline AUC | Age/Sex Baseline AUC |
| :--- | :---: | :---: | :---: | :---: |
| **KKI** ($n = 83$) | **0.4694** (`0.469449`) | `[0.3353, 0.6088]` | `0.6118` | `0.4832` |
| **NeuroIMAGE** ($n = 48$) | **0.5757** (`0.575652`) | `[0.4243, 0.7322]` | `0.7061` | `0.6348` |
| **OHSU** ($n = 79$) | **0.6042** (`0.604247`) | `[0.4755, 0.7239]` | `0.6461` | `0.6338` |
| **Peking** ($n = 194$) | **0.6732** (`0.673187`) | `[0.5985, 0.7477]` | `0.6112` | `0.6755` |

### 3.3. Section 8.1 Development Success Criteria (`no_exclusion_404`)
- **Criterion 1** ($p_{\text{perm}} < 0.05$): **MET** (`0.0050 < 0.05`).
- **Criterion 2** ($\text{CI}_{95\%,\text{lo}}[\Delta\text{AUC}_{\text{motion}}] > 0$): **NOT MET** (`-0.0890 <= 0`).
- **Combined Section 8.1 Criterion:** **NOT MET** (`meets_dev_success_criteria = False`).

---

## 4. Reading Under the Registered Rules (2026-11-18)

- **Primary Analysis ($N = 368$, Registered Section 5.3 Exclusion):** Pooled LOIO ROC-AUC is `0.5241` (95% CI `[0.4691, 0.5797]`), within-institution permutation $p_{\text{perm}} = 0.2298$ (not $< 0.05$), and $\Delta\text{AUC}_{\text{motion}} = -0.0797$ (95% CI `[-0.1520, -0.0099]`, lower bound not $> 0$). Neither Section 8.1 criterion is met.
- **Sensitivity Analysis ($N = 404$, Registered Section 5.4 No-Exclusion Analysis):** Pooled LOIO ROC-AUC is `0.5740` (95% CI `[0.5243, 0.6243]`) with $p_{\text{perm}} = 0.0050$ ($< 0.05$), while motion-only baseline AUC (`0.6158`) exceeds fMRI AUC (`0.5740`), yielding $\Delta\text{AUC}_{\text{motion}} = -0.0418$ (95% CI `[-0.0890, +0.0087]`, lower bound not $> 0$). Section 8.1 criterion 2 is not met, so combined Section 8.1 criteria are not met.
- **Protocol Conformance:** No parameters, seeds, features, ROI definitions, or model choices are modified in response to these development results.
