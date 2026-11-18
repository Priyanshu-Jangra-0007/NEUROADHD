# Study 2 Development Feature Set Plan (Design Only — Not Executed)

**Date:** 2026-11-18  
**Status:** **PLANNED ONLY — ZERO TIME SERIES EXTRACTED OR COMPUTED**  
**Branch:** `feat/study2-athena-acquisition`  
**Preregistration Reference:** [`docs/study2/PREREGISTRATION.md`](PREREGISTRATION.md) (SHA-256: `e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`)  
**Frozen Cohort Reference:** [`docs/study2/COHORT_COUNTS_2026-11-18.md`](COHORT_COUNTS_2026-11-18.md)  
**Deviation Reference:** [`docs/study2/DEVIATIONS.md`](DEVIATIONS.md) (Branch b — NIAK volume-centered FD, first 3 frames removed; no-exclusion elevated to co-primary sensitivity)

---

## 1. Exact Files Read (Development Cohort Only — Zero NYU Access)

| Input File | Purpose | Strict Filter / Boundary |
| :--- | :--- | :--- |
| `NeuroADHD/data/athena/staging/ADHD200_CC200_TCs_filtfix.tar.gz` | Primary band-pass filtered ($0.009\text{--}0.08\text{ Hz}$), non-GSR CC200 BOLD time courses | Stream/extract **ONLY** `<Site>/<ID>/sfnwmrda<ID>_session_1_rest_1_cc200_TCs.1D` for `<Site> ∈ {KKI, NeuroIMAGE, OHSU, Peking_1, Peking_2, Peking_3}` ($N = 404$). **Skip `NYU/`, `WashU/`, `Pittsburgh/`, unfiltered `snwmrda*`, and `rest_2`/`rest_3`/`rest_4` completely.** |
| `NeuroADHD/data/athena/motion/dev_niak_fd_metrics.csv` | Blinded per-scan motion metrics and exclusion flags (`mean_fd_mm`, `max_disp_mm`, `exclude_primary_050`, `exclude_strict_020`, `frames_cc200`) | Contains only the 404 development `rest_1` scans (zero NYU, zero diagnosis column). |
| `NeuroADHD/data/athena/phenotypes/adhd200_preprocessed_phenotypics.tsv` | Binary `DX` (`0` $\to$ TDC `0`; `{1, 2, 3}` $\to$ ADHD `1`), `Age`, and `Gender` for development cohort and demographic baseline | Filter strictly to the 404 matched development `ScanDir ID`s. Zero NYU imaging or connectivity joined. |

- **Planned Output Location (Git-Ignored under `**/athena/`):** `NeuroADHD/data/athena/features/` (npz/parquet matrices + `valid_roi_indices.json` + summary metadata with zero subject IDs printed to logs).

---

## 2. Pre-Registered ROI Rule (`PREREGISTRATION.md`, Section 5.1)

1. **`.1D` Parsing:**
   - Parse each Athena CC200 `.1D` file as tab/whitespace-delimited table.
   - Verify the header row (`File`, `Sub-brick`, followed by `Mean_<roi_id>` columns) and drop the 2 leading non-ROI metadata columns (`File`, `Sub-brick`) to obtain $X_i \in \mathbb{R}^{T_i \times K_{\text{raw}}}$.
   - Verify $K_{\text{raw}}$ (expected 190 valid CC200 parcels in Athena CC200 TCs) and column headers are identical across all 404 development scans.
2. **Development-Only Non-Zero Variance Rule (Unified Across All 404 Dev Scans):**
   - Compute per-ROI temporal standard deviation $\sigma_{i,k} = \operatorname{std}(X_i[:, k])$ across time for every development subject $i \in \{1, \dots, 404\}$.
   - Keep ROI $k$ **if and only if** $\sigma_{i,k} > 0$ and all values in $X_i[:, k]$ are finite across **100% of all 404 development subjects**.
   - Both the **primary exclusion analysis ($N = 368$)** and the **co-primary no-exclusion analysis ($N = 404$)** use this single unified, frozen ROI set (per [`DEVIATIONS.md`](DEVIATIONS.md) Section 3.5; strict exclusion is descriptive only and not run as LOIO).
   - Drop any ROI that is constant ($\sigma_{i,k} = 0$) or contains NaN/Inf in even one development subject.
   - Freeze and record the retained ROI column names and indices (`K` kept, `K_raw - K` dropped).
3. **Pre-Specified Sealed-Site ROI Rule (Deferred Until Final Scoring):**
   - The sealed site (NYU) **never** modifies the development ROI set.
   - If any kept development ROI is constant or missing in an NYU subject at final one-shot evaluation, all Fisher-$z$ connectivity edges touching that ROI for that subject are set to `0.0` (prior mean $r = 0, z = 0$) without re-selecting features or refitting weights.

---

## 3. Pairwise Pearson Correlation & Fisher $z$ Edges (`PREREGISTRATION.md`, Section 2)

For each subject $i$ with sliced time series $X_i^{\text{valid}} \in \mathbb{R}^{T_i \times K}$:
1. Compute the $K \times K$ sample Pearson correlation matrix:
   $$r_{jk} = \frac{\sum_{t=1}^{T_i} (x_{jt} - \bar{x}_j)(x_{kt} - \bar{x}_k)}{\sqrt{\sum_{t=1}^{T_i} (x_{jt} - \bar{x}_j)^2 \sum_{t=1}^{T_i} (x_{kt} - \bar{x}_k)^2}}$$
2. Clip off-diagonal entries to $[-1 + \epsilon, 1 - \epsilon]$ with $\epsilon = 10^{-7}$ (`np.clip(R, -0.9999999, 0.9999999)`) to prevent numerical $\pm\infty$ in $\operatorname{arctanh}$.
3. Apply Fisher's $r$-to-$z$ transform:
   $$z_{jk} = \frac{1}{2} \ln\left(\frac{1 + r_{jk}}{1 - r_{jk}}\right) = \operatorname{arctanh}(r_{jk})$$
4. Vectorize the strict upper triangle (`np.triu_indices(K, k=1)`) into a 1D vector of length $P = \frac{K(K - 1)}{2}$ (`float64`).
5. Stack across subjects into $Z \in \mathbb{R}^{N \times P}$.

---

## 4. Mandatory Shape, Integrity & Anti-Leakage Checks

Every extraction run must assert all of the following before writing output files:

1. **Archive Filtering Check:** Assert 0 members from `NYU`, `WashU`, or `Pittsburgh` are read or extracted; assert exactly 404 development `rest_1` `.1D` files are parsed.
2. **Time-Series Shape Check:** For every scan $i$, assert $X_i.\text{shape}[0] == \text{frames\_cc200}_i$ (matching `dev_niak_fd_metrics.csv`, i.e., KKI 121/149, NeuroIMAGE 258, OHSU 75/73, Peking 233) and $X_i.\text{shape}[1] == K_{\text{raw}}$ identical for all 404 scans.
3. **ROI Variance Check:** Assert $\min_{i, k \in \text{valid}} \sigma_{i,k} > 0$ and `np.isfinite(X_valid).all()`.
4. **Correlation Matrix Checks:** For every subject $i$, assert:
   - `R.shape == (K, K)`
   - Symmetry: `max(abs(R - R.T)) < 1e-12`
   - Unit diagonal: `max(abs(diag(R) - 1.0)) < 1e-12`
   - Bounded range: `R.min() >= -1.0 - 1e-12` and `R.max() <= 1.0 + 1e-12`
5. **Edge Matrix Checks:** Assert `Z.shape == (N, K * (K - 1) // 2)`, `Z.dtype == np.float64`, `0` NaNs, and `0` Infs.
6. **Cohort Alignment Check:** Assert per-site and total counts match [`COHORT_COUNTS_2026-11-18.md`](COHORT_COUNTS_2026-11-18.md) exactly (All/No-Exclusion: 404 [242 TDC / 162 ADHD]; Primary: 368 [230 TDC / 138 ADHD]; Strict: 133 [98 TDC / 35 ADHD]).
7. **Deterministic Reproducibility & Privacy Checks:** Identical output checksum across repeated runs; zero individual subject IDs printed to stdout/logs.
8. **Zero-Leakage Verification (At Modeling Time):** All scaling, imputation, ANOVA top-100 edge selection (`SelectKBest`), and Random Forest fitting (`n_estimators=50, max_depth=3, class_weight='balanced'`) remain strictly inside each LOIO training fold.

---

## 5. Final Sealed-Site (NYU) Motion & Scoring Prerequisite

- **NYU Remains Sealed Throughout Development:** No NYU BOLD time series, motion traces, or connectivity edges are downloaded, extracted, or inspected during development feature generation or LOIO cross-validation.
- **Final-Step NYU Motion Prerequisite:** At the final one-shot external validation step (only after all development LOIO results are locked and reported), **NYU motion data will be needed** to apply the matching motion exclusion rule and score the motion-only baseline on NYU.
- **Label-Blind NYU Motion Procedure:** Under `DEVIATIONS.md` Branch (b), `intermediate_nyu.tar` (NITRC Release 1600 `/frs/downloadlink.php/3808`, ~263 MB) will be obtained at that final stage, and NYU FD will be computed **strictly label-blind** (same volume-centered, first-3-frames-dropped Power et al. 2012 FD pipeline, with zero NYU diagnosis labels joined during FD calculation or threshold checking) prior to the single frozen-model scoring run on NYU.
