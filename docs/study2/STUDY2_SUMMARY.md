# Study 2 Final Summary: Multi-Site Resting-State fMRI Generalization (Athena CC200)

**Date:** 2026-11-18  
**Branch:** `feat/study2-athena-acquisition`  
**Locked Protocol File (Unmodified):** [`docs/study2/PREREGISTRATION.md`](PREREGISTRATION.md)  
**Locked Protocol Commit:** `2d7ed3b3be3780c0b8a5b3de7d3c786bb77df19e` (`2d7ed3b`)  
**Locked Protocol SHA-256:** `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`  
**Pipeline:** Locked Study 1 pipeline (`SimpleImputer(strategy='median')` → `VarianceThreshold(0.0)` → `StandardScaler()` → `SelectKBest(f_classif, k=100)` → `RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42)`), refitted inside each training fold on 17,955 Fisher-$z$ edges from 190 CC200 ROIs.

> All numbers in this document are drawn solely from previously recorded Study 2 outputs ([`COHORT_COUNTS_2026-11-18.md`](COHORT_COUNTS_2026-11-18.md), [`DEV_RESULTS_2026-11-18.md`](DEV_RESULTS_2026-11-18.md), [`SEALED_RESULTS_2026-11-18.md`](SEALED_RESULTS_2026-11-18.md), [`DEVIATIONS.md`](DEVIATIONS.md), and [`OSF_RECORD.md`](OSF_RECORD.md)). No new classifier analyses were run and no parameters, seeds, features, or ROI definitions were modified after inspecting results.

---

## 1. OSF Registrations & Chronological Git Audit Trail

### 1.1. Public OSF Registrations (All Timestamps UTC)
1. **`PREREGISTRATION.md` uploaded to OSF:** `2026-11-17 14:52 UTC`
2. **Open-Ended Registration (registered first; metadata & hashes):** [https://osf.io/db7nv](https://osf.io/db7nv) — DOI `10.17605/OSF.IO/DB7NV` (`2026-11-17 14:56 UTC`, Public, CC0 1.0)
3. **Primary OSF Preregistration (registered second; complete binding protocol):** [https://osf.io/grbt2](https://osf.io/grbt2) — DOI `10.17605/OSF.IO/GRBT2`, `2026-11-17`, approximately 14:58 UTC (inferred from the OSF page display), Public, CC0 1.0

### 1.2. Chronological Annotated Git Tags (pushed to origin)
All four milestone tags lie in strict chronological order along `HEAD`'s ancestry, and the OSF registration commit `2d7ed3b` is a verified ancestor of every tag:

| Order | Tag Name | Target Commit | Timestamp (UTC) | Milestone |
| :-: | :--- | :---: | :---: | :--- |
| 0 | *(OSF registration commit)* | `2d7ed3b` | `2026-11-17` | Frozen `PREREGISTRATION.md` committed prior to data acquisition |
| 1 | `study2-dev-analysis-frozen` | `3474758` | `2026-11-18T06:52:18Z` | Development script (`6e812b7`), tests, and feature `.npz` files frozen before real-label run |
| 2 | `study2-dev-results` | `c9a5ca7` | `2026-11-18T07:04:48Z` | Single development LOIO run recorded (`DEV_RESULTS_2026-11-18.md`) |
| 3 | `study2-sealed-frozen` | `a75d6b4` | `2026-11-18T07:13:05Z` | Sealed-site script (`b96d1ed`) and tests frozen before NYU archive acquisition or opening |
| 4 | `study2-sealed-results` | `65baffc` | `2026-11-18T07:23:56Z` | Single one-shot sealed NYU run recorded (`SEALED_RESULTS_2026-11-18.md`) |

- **Development results artifact SHA-256** (`study2_dev_results.json` and `study2_dev_run.log`, separate files, Git-ignored): `6d072448bf6573cd7f6e0bd65c0b6a141bc2cf60f2ea0e4a69bcc504918dc0d5`
- **Sealed results artifact SHA-256** (`study2_sealed_results.json` and `study2_sealed_run.log`, separate files, Git-ignored): `117e0395e73f9c9e0388d9a37d6e7de1405c052cfdc2c35769891cbc7d548dc2`

---

## 2. Cohort Summary (Aggregate Counts Only)

| Cohort / Regime | Total ($N$) | TDC | ADHD | Excluded | Attrition Fisher Exact $p$ (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Development Sensitivity (`no_exclusion_404`, Section 5.4)** | **404** | 242 | 162 | 0 | — |
| **Development Primary (`primary_368`, Section 5.3 FD rule)** | **368** | 230 | 138 | 36 (12 TDC [5.0%], 24 ADHD [14.8%]) | `0.0011` |
| **Development Strict (`>0.20 mm` / `>10%`, descriptive only)** | **133** | 98 | 35 | 271 (144 TDC [59.5%], 127 ADHD [78.4%]) | `0.0001` |
| **Sealed NYU Sensitivity (`sensitivity_no_exclusion`, Section 5.4)** | **216** | 98 | 118 | 0 | — |
| **Sealed NYU Primary (`primary_sealed`, Section 5.3 FD rule)** | **196** | 91 | 105 | 20 (7 TDC [7.1%], 13 ADHD [11.0%]) | `0.3568` |

- Development LOIO institutions (`primary_368` / `no_exclusion_404`): **KKI** (`79` [60/19] / `83` [61/22]), **NeuroIMAGE** (`42` [23/19] / `48` [23/25]), **OHSU** (`58` [33/25] / `79` [42/37]), **Peking (1–3 combined)** (`189` [114/75] / `194` [116/78]).
- Sealed NYU scans in `ADHD200_CC200_TCs_filtfix.tar.gz`: `216` complete `rest_1` scans (the `263` rows in `adhd200_preprocessed_phenotypics.tsv` include `47` test-release rows absent from the CC200 archive and were not scored). Zero NYU ROIs required Section 5.1 zero-filling (`0` scans / `0` ROI instances).

---

## 3. Registered Hypotheses (H1–H4) & Outcomes Under the Registered Rules

### 3.1. Summary of Hypotheses H1–H4 Across All Four Scored Regimes

| Hypothesis | Registered Rule (`PREREGISTRATION.md`) | Regime | Estimate & 95% Stratified Bootstrap CI ($B = 2,000$) | Test / Criterion | Outcome Under Registered Rules |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **H1: Development Generalization** | Pooled LOIO ROC-AUC exceeds chance: within-institution permutation $p_{\text{perm}} < 0.05$ ($B = 1,000$; Section 7.1 & 8.1 Criterion 1) | Dev Primary ($N = 368$) | `0.5241` `[0.4691, 0.5797]` | `p_perm = 0.2298` (`229/1000` exceed; null mean `0.4924`, SD `0.0399`) | **NOT MET** |
| **H1: Development Generalization** | Same rule applied to registered Section 5.4 sensitivity analysis | Dev Sensitivity ($N = 404$) | `0.5740` `[0.5243, 0.6243]` | `p_perm = 0.0050` (`4/1000` exceed; null mean `0.4830`, SD `0.0359`) | **MET** *(Criterion 1 only; combined Section 8.1 NOT MET)* |
| **H2: Outperforming Motion-Only Baseline** | Paired bootstrap 95% CI lower bound for $\Delta\text{AUC}_{\text{motion}} = \text{AUC}_{\text{fMRI}} - \text{AUC}_{\text{motion}} > 0$ (Section 6 & 8.1 Criterion 2) | Dev Primary ($N = 368$) | `-0.0797` `[-0.1520, -0.0099]` *(fMRI `0.5241` vs Motion `0.6038`)* | $\text{CI}_{\text{lo}} = -0.1520 \le 0$ | **NOT MET** |
| **H2: Outperforming Motion-Only Baseline** | Same rule applied to registered Section 5.4 sensitivity analysis | Dev Sensitivity ($N = 404$) | `-0.0418` `[-0.0890, +0.0087]` *(fMRI `0.5740` vs Motion `0.6158`)* | $\text{CI}_{\text{lo}} = -0.0890 \le 0$ | **NOT MET** |
| **H3: Incremental Value Over Age & Sex Baseline** | Paired bootstrap $\Delta\text{AUC}_{\text{age/sex}} = \text{AUC}_{\text{fMRI}} - \text{AUC}_{\text{Age/Sex}}$ (Section 7.2 secondary endpoint) | Dev Primary ($N = 368$) | `-0.0921` `[-0.1649, -0.0169]` *(fMRI `0.5241` vs Age/Sex `0.6163`)* | $\text{CI}_{\text{lo}} = -0.1649 \le 0$ | **NOT MET** |
| **H3: Incremental Value Over Age & Sex Baseline** | Same rule applied to registered Section 5.4 sensitivity analysis | Dev Sensitivity ($N = 404$) | `-0.0116` `[-0.0795, +0.0568]` *(fMRI `0.5740` vs Age/Sex `0.5857`)* | $\text{CI}_{\text{lo}} = -0.0795 \le 0$ | **NOT MET** |
| **H4: Sealed External Generalization (NYU)** | Single scoring of pooled-development model on sealed NYU; Confirmed Generalization requires 95% bootstrap CI lower bound $> 0.50$ (Section 4.3 & 8.2) | **Sealed Primary ($N = 196$, Governing)** | **`0.5765`** **`[0.4962, 0.6565]`** (`[0.496177, 0.656518]`) | Point `> 0.50`, $\text{CI}_{\text{lo}} = 0.496177 \le 0.50$ | **Inconclusive Zone** (`inconclusive`) |
| **H4: Sealed External Generalization (NYU)** | Same rule applied to registered Section 5.4 sensitivity analysis | Sealed Sensitivity ($N = 216$) | `0.5770` `[0.5000, 0.6528]` (`[0.499996, 0.652819]`) | Point `> 0.50`, $\text{CI}_{\text{lo}} = 0.499996 \le 0.50$ | **Inconclusive Zone** (`inconclusive`) |

- **Combined Section 8.1 Development Success Criteria:** **NOT MET** in both `primary_368` (neither criterion met) and `no_exclusion_404` (Criterion 1 met, Criterion 2 not met because motion-only baseline AUC `0.6158` exceeded fMRI AUC `0.5740`).

### 3.2. Baseline & Per-Institution Reference Numbers

- **Development Baselines (Pooled LOIO ROC-AUC [95% CI]):**
  - `primary_368` ($N = 368$): Motion-only `0.6038 [0.5508, 0.6585]`; Age & Sex `0.6163 [0.5650, 0.6672]`.
  - `no_exclusion_404` ($N = 404$): Motion-only `0.6158 [0.5670, 0.6644]`; Age & Sex `0.5857 [0.5369, 0.6363]`.
- **Development Per-Institution fMRI LOIO ROC-AUCs [95% CI]:**
  - `primary_368`: KKI `0.5500 [0.3930, 0.7026]`, NeuroIMAGE `0.5698 [0.3959, 0.7529]`, OHSU `0.6048 [0.4521, 0.7430]`, Peking `0.5346 [0.4537, 0.6152]`.
  - `no_exclusion_404`: KKI `0.4694 [0.3353, 0.6088]`, NeuroIMAGE `0.5757 [0.4243, 0.7322]`, OHSU `0.6042 [0.4755, 0.7239]`, Peking `0.6732 [0.5985, 0.7477]`.
- **Sealed NYU Descriptive Baselines (ROC-AUC [95% CI], descriptive only):**
  - `primary_sealed` ($N = 196$): Motion-only `0.6069 [0.5276, 0.6805]`; Age & Sex `0.5806 [0.5000, 0.6593]` (`[0.500044, 0.659267]`).
  - `sensitivity_no_exclusion` ($N = 216$): Motion-only `0.6076 [0.5335, 0.6825]`; Age & Sex `0.6150 [0.5367, 0.6928]`.

---

## 4. Section 8.2 Outcome Zone (Worded Exactly as Registered)

Under `PREREGISTRATION.md` Section 8.2, both the governing sealed primary result (`primary_sealed`, $N = 196$: `0.576452`, 95% CI `[0.496177, 0.656518]`) and the sealed sensitivity result (`sensitivity_no_exclusion`, $N = 216$: `0.576963`, 95% CI `[0.499996, 0.652819]`) satisfy the condition *Point estimate $> 0.50$, but 95% bootstrap CI lower bound $\le 0.50$* and fall in the **Inconclusive Zone**. Quoted verbatim from [`docs/study2/PREREGISTRATION.md`](PREREGISTRATION.md#L159):

> | **Inconclusive Zone** | Point estimate $> 0.50$, but 95% bootstrap CI lower bound $\le 0.50$ | **Results are statistically ambiguous.** While a positive discrimination point estimate is observed, the confidence interval spans chance ($0.50$). Due to the sample size constraint (the 95% CI half-width is approximately $\pm 0.07$ at $N \approx 216$), statistical power is insufficient to confirm discrimination, and external validity cannot be claimed. |

---

## 5. Complete Record of Protocol Deviations & Clarifications (`DEVIATIONS.md`)

All deviations and clarifications were logged in [`docs/study2/DEVIATIONS.md`](DEVIATIONS.md) without modifying `PREREGISTRATION.md`:

### 5.1. Protocol Deviations
1. **NIAK Rigid-Body Transforms Used Instead of AFNI `3dvolreg` `rp_*.1D` (Registration Section 5.2):** Because the six development `<Site>_preproc_xfms.tar` archives contained FSL FNIRT warp fields only (`0 / 404` AFNI `rp_*.1D` files; commit `08f87fe`), the pre-inspection fallback rule (Branch b, committed prior to opening the archives in `ac517ce`) was activated. Power et al. (2012) FD ($r = 50\text{ mm}$, $\text{FD}_0 = 0$) was computed from Neuro Bureau NIAK $4 \times 4 \times T_{\text{NIAK}}$ rigid-body transformation matrices with the **first 3 frames removed** (`DUMMY_FRAMES_DROPPED = 3` in `scripts/compute_athena_fd.py` and `scripts/run_study2_sealed_site.py`, retaining $T_{\text{NIAK}} - 3$ motion frames). As documented in `DEVIATIONS.md` Section 4.5, each Athena CC200 `.1D` file contains **$T_{\text{BOLD}} = T_{\text{NIAK}} - 4$ BOLD data rows** (Athena's 4-volume removal `3dcalc '[4..$]'`) plus **1 header row** (`T_BOLD + 1 = T_NIAK - 3` total non-empty lines across all 404 development scans), so the frozen FD calculator dropped 3 leading NIAK frames rather than 4 (retaining 1 extra leading NIAK frame with $\text{FD}_0 = 0.0$ before the $T_{\text{BOLD}}$ BOLD data rows). The same motion source was used for the motion-only baseline.
2. **FD Calculator Correction via NIfTI Volume-Centre Conjugation (`39 → 36` Primary Exclusions; Pre-Label-Join):** Strictly label-free validation against Athena `*_motion.csv` extrema ([`NIAK_VS_ATHENA_MOTION_VALIDATION.md`](NIAK_VS_ATHENA_MOTION_VALIDATION.md)) identified spurious lever-arm translation $(I - R)\mathbf{c}$ when reading world-origin `transf[:3, 3]` on off-origin NIfTI grids (NeuroIMAGE $\rho = 0.5339$, median ratio `1.4619`). Conjugating each scan's transform to its NIfTI volume geometric center via $M_{\text{centered}}(t) = T(-\mathbf{c}) M(t) T(+\mathbf{c})$ (commit `3c2b7ac`, before any diagnosis join) raised pooled translation Spearman $\rho$ from `0.8757` to `0.9704` (NeuroIMAGE `0.5339` to `0.8809`) and changed blinded primary exclusions from **39 to 36** (retained `365 → 368`; strict exclusions `299 → 271`). Earlier uncentered FD tables from `08f87fe` are superseded.
3. **NeuroIMAGE Registration-Failure Scan (~86° NIAK Divergence):** One NeuroIMAGE development scan exhibited a NIAK `minctracc` registration failure ($\max |\theta| = 85.98^\circ$ on 203 of 258 retained frames vs. `7.34°` in Athena `NeuroIMAGE_motion.csv`). Per the pre-specified disposition (no ad-hoc removal), this scan was excluded under the primary motion rule ($N = 368$) and retained in the registered Section 5.4 no-exclusion sensitivity analysis ($N = 404$).
4. **Strict Motion Threshold (`>0.20 mm` OR `>10%` Frames) Descriptive Only:** Under the strict threshold, 3 of 4 LOIO institution folds dropped below 15 subjects per class (`KKI`: ADHD = 2; `NeuroIMAGE`: TDC = 11, ADHD = 8; `OHSU`: TDC = 13, ADHD = 6). It was therefore retained for descriptive attrition reporting only and was not run as a LOIO classification model.
5. **Wording Fix ("co-primary" → Primary vs. Section 5.4 Sensitivity, Fixed Before Any Real-Label Analysis):** Earlier wording in `DEVIATIONS.md` (`ac517ce`) referred to the no-exclusion analysis as "co-primary." Prior to any real-label analysis (commit `3063b28`), this was corrected to match `PREREGISTRATION.md`: the **primary analysis** is pooled LOIO ROC-AUC at $N = 368$ (Section 5.3 exclusion), the **no-exclusion analysis** at $N = 404$ is the registered **Section 5.4 sensitivity analysis**, and the $N = 404$ cited in Section 7.1 is the preliminary pre-exclusion count from Section 4.2.
6. **Post-Freeze Staging of `intermediate_nyu.tar`:** Per [`SEALED_RUN_PLAN.md`](SEALED_RUN_PLAN.md) (`837b33c`), `intermediate_nyu.tar` (NITRC Release 1600) was downloaded after tag `study2-sealed-frozen` (`a75d6b4`) and copied to Git-ignored `NeuroADHD/data/athena/staging/intermediate_nyu.tar` (commit `f21baa1`; size `275,527,680` bytes, self-computed SHA-256 `9640ee0a1623efe91d9eab9ad6a4024d240cdfad3886ecf9b1124a778b141eb4`; non-extracting TAR check passed with `2,665` members / `1,368` files). Zero NYU data were opened or computed before the single sealed execution.

### 5.2. Clarifications (Not Deviations)
1. **190 ROIs Retained (10 Empty Clusters in CC200 Template; Identical Across $N = 404$ and $N = 368$):** Direct inspection of `templates/ADHD200_parcellate_200.nii.gz` inside `ADHD200_CC200_TCs_filtfix.tar.gz` confirmed exactly **190** populated non-zero ROI labels in `[1, 200]`; 10 cluster IDs (`2, 5, 11, 25, 34, 68, 82, 95, 126, 191`) have 0 voxels and are absent from `3dROIstats` `.1D` output. Across all 404 development scans, **190 / 190** ROIs had finite, non-zero temporal variance (minimum `std = 0.004031`, **0 dropped**), yielding $P = 190 \times 189 / 2 = 17,955$ edges and an identical ROI set for both $N = 404$ and $N = 368$.
2. **Pearson $r$ Clipping Before Fisher $z$:** Off-diagonal correlations were clipped to `[-1 + 1e-7, 1 - 1e-7]` (`FISHER_CLIP_EPS = 1e-7`) prior to $\operatorname{arctanh}(r)$ solely for numerical stability.
3. **Exact Definition of "Max Displacement" Used by the Scripts:** Quoted verbatim from `scripts/compute_athena_fd.py` (lines 292–294), computed from the volume-centered, first-3-frames-dropped NIAK translations `params[:, 3:6]` (mm) and paired with `mean_fd_mm` in `motion_X`:
   ```python
   # Relative translation from first retained frame (Euclidean mm)
   rel_trans = params[:, 3:6] - params[0, 3:6]
   max_disp_mm = float(np.max(np.linalg.norm(rel_trans, axis=1)))
   ```

Additional label-free checks and their counts are logged in DEVIATIONS.md.

---

## 6. Prior-Exposure Disclosures (`OSF_RECORD.md`)

1. **Study 1 Subject Overlap:** **126** Study 2 development subjects were previously part of the Study 1 exploratory 153-subject C-PAC cohort where the locked pipeline's architecture and hyperparameters were selected: **KKI** (`24`), **NeuroIMAGE** (`19`), **Peking 1–3** (`69`), and **OHSU** (`14` controls). In addition, **6 NYU subjects** were present in Study 1's C-PAC cohort.
2. **Correction to OSF Foreknowledge Declaration:** The OSF registration form selected the prior-observation option *"Authors' limited observation of the data could not influence their analysis decisions."* As explicitly documented in [`OSF_RECORD.md`](OSF_RECORD.md), that option **understates** the actual prior exposure because those 126 development subjects (and 6 NYU subjects) were in Study 1 where the locked pipeline hyperparameters were chosen (under C-PAC rather than Athena).
3. **Zero Athena Exposure at Registration:** No Athena derivatives (`.1D` time series, motion traces, or connectivity matrices) had been downloaded, unpacked, inspected, or analyzed at the time of either 2026-11-17 OSF registration or commit `2d7ed3b`.

---

## 7. Limitations

1. **Differential Motion Attrition & Regime Differences:** Head motion differed substantially by diagnosis in the development cohort. Under the primary FD threshold (`>0.50 mm` / `>20%`), ADHD scans were excluded at nearly three times the TDC rate (`14.8%` vs. `5.0%`, Fisher $p = 0.0011$; NeuroIMAGE `24.0%` vs. `0.0%`, $p = 0.0227$). Under the strict threshold, attrition reached `78.4%` ADHD vs. `59.5%` TDC ($p = 0.0001$), collapsing 3 of 4 LOIO folds below 15 subjects per class. Removing the 36 highest-motion development scans shifted fMRI pooled LOIO AUC from `0.5740` ($N = 404$, `p_perm = 0.0050`) to `0.5241` ($N = 368$, `p_perm = 0.2298`), with Peking shifting from `0.6732` to `0.5346`. AUC and permutation results differed between the N=404 and N=368 regimes; the cause was not tested.
2. **Study 1 Subject Overlap & Development Optimism:** The locked pipeline's hyperparameters were chosen in Study 1 on a cohort that included 126 of the development subjects, so development LOIO estimates may be optimistic for those subjects; only the sealed NYU result is close to independent (6 subjects overlapped).
3. **Non-Imaging Baselines Match or Exceed fMRI AUC Across Every Regime:** Across all four scored regimes, both simple 2-feature non-imaging logistic baselines achieved point estimates higher than the locked fMRI pipeline:
   - Development Primary ($N = 368$): fMRI `0.5241` vs. Motion `0.6038` ($\Delta = -0.0797$ `[-0.1520, -0.0099]`) and Age/Sex `0.6163` ($\Delta = -0.0921$ `[-0.1649, -0.0169]`).
   - Development Sensitivity ($N = 404$): fMRI `0.5740` vs. Motion `0.6158` ($\Delta = -0.0418$ `[-0.0890, +0.0087]`) and Age/Sex `0.5857` ($\Delta = -0.0116$ `[-0.0795, +0.0568]`).
   - Sealed NYU Primary ($N = 196$, descriptive baselines): fMRI `0.5765` vs. Motion `0.6069` and Age/Sex `0.5806`.
   - Sealed NYU Sensitivity ($N = 216$, descriptive baselines): fMRI `0.5770` vs. Motion `0.6076` and Age/Sex `0.6150`.
4. **Confidence Interval Width & Statistical Power:** On the sealed NYU cohort, the observed 95% stratified bootstrap CI half-width was approximately $\pm 0.08$ (`[0.496177, 0.656518]` at $N = 196$, width `0.1603`; `[0.499996, 0.652819]` at $N = 216$, width `0.1528`), spanning chance (`0.50`) in both regimes. As noted in `PREREGISTRATION.md` Section 7.2, statistical power for detecting a moderate incremental $\Delta\text{AUC} = +0.04$ over age and sex is approximately `18%` at $N = 404$ and `26%` at $N = 620$; accordingly, wide intervals and non-significant differences reflect finite-sample precision limits alongside the lack of demonstrated incremental fMRI signal under this locked pipeline.
5. **Cross-Pipeline Motion Substitution & Atlas Divergence:** FD was computed from NIAK rigid-body transforms rather than Athena-native AFNI `3dvolreg` `rp_*.1D` traces (validated label-free at pooled $\rho = 0.9704$ translation / $\rho = 0.9730$ rotation $L_2$ against Athena extrema, with one NeuroIMAGE NIAK registration failure). Per `PREREGISTRATION.md` Section 9, the Athena AFNI-space CC200 parcellation cannot be assumed identical to Study 1's C-PAC FSL-space CC200 parcellation, so Study 2 is not directly numerically comparable to Study 1.
