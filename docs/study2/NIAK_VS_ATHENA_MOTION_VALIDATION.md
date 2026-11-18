# Label-Free Validation: NIAK-Derived Motion vs. Athena AFNI Extrema (Study 2)

**Date:** 2026-11-18  
**Branch:** `feat/study2-athena-acquisition`  
**Script:** [`scripts/validate_niak_vs_athena_motion.py`](../../scripts/validate_niak_vs_athena_motion.py)  
**Guardrails:** Zero diagnosis labels loaded or joined; zero NYU data accessed; zero individual subject IDs logged or printed.

---

## 1. Athena Site `*_motion.csv` Schema (Columns & Units)

All six development-site motion summary CSVs (`KKI_motion.csv` [83 rows], `NeuroIMAGE_motion.csv` [48 rows], `OHSU_motion.csv` [237 rows across runs; 79 `rest_1`], `Peking_1_motion.csv` [85 rows], `Peking_2_motion.csv` [67 rows], `Peking_3_motion.csv` [42 rows]) share the identical 11-column header produced by Neuro Bureau's `est_motion_afni.pl`:

| Index | Exact Column Name | Units | Description |
| :-: | :--- | :---: | :--- |
| 0 | `File` | string | Source AFNI trace filename (`rp_<ID>_session_1_rest_<R>.1D`) |
| 1 | `Max Motion (mm)` | **mm** | Non-negative scalar peak translational displacement across the run |
| 2 | `Max Motion Time Point` | TR index | Time-point index where `Max Motion (mm)` occurred |
| 3 | ` Max Rotation (degree)` | **degrees** | Non-negative scalar peak angular rotation across the run |
| 4 | ` Max Rotation Time Point` | TR index | Time-point index where `Max Rotation (degree)` occurred |
| 5 | ` Max X (mm)` | **mm** | Signed maximum translation along X (R–L) axis |
| 6 | ` Max Y (mm)` | **mm** | Signed maximum translation along Y (A–P) axis |
| 7 | ` Max Z (mm)` | **mm** | Signed maximum translation along Z (I–S) axis |
| 8 | ` Max Roll (degree)` | **degrees** | Signed maximum roll rotation |
| 9 | ` Max Pitch (degree)` | **degrees** | Signed maximum pitch rotation |
| 10 | ` Max Yaw (degree)` | **degrees** | Signed maximum yaw rotation |

*(Note: Columns 3–10 carry a single leading space in the upstream NITRC CSV header.)*

---

## 2. Critical Physics Finding Caught by Validation: NIfTI World-Origin Lever-Arm Effect

Initial naive extraction of `transf[:3, 3]` at the world coordinate origin `(0, 0, 0)` revealed a severe translation discrepancy on NeuroIMAGE ($\rho = 0.5339$, median ratio $= 1.4619$). Inspection of the NIfTI headers of `motion_target_*_session_1_run1.nii.gz` revealed the physical cause:
- **KKI:** volume geometric center at `[0.0, 0.0, 0.0] mm` (offset $0.0\text{ mm}$)
- **OHSU:** median volume center at `[5.4, 13.7, -19.0] mm` (offset $24.0\text{ mm}$)
- **Peking:** median volume center at `[1.6, 17.2, 21.0] mm` (offset $27.2\text{ mm}$)
- **NeuroIMAGE:** median volume center at `[4.1, 53.3, -10.7] mm` (offset **$54.5\text{ mm}$**)

Because MINC `minctracc -est_center` rotates about the brain center $\mathbf{c}$ and writes world-space matrix $M(\mathbf{x}) = R(\mathbf{x} - \mathbf{c}) + \mathbf{c} + \mathbf{t}_{\text{center}} = R\mathbf{x} + [\mathbf{t}_{\text{center}} + (I - R)\mathbf{c}]$, reading the 4th column directly injected spurious lever-arm translation $(I - R)\mathbf{c}$ whenever $\mathbf{c} \ne \mathbf{0}$. Conjugating each matrix to the scan's NIfTI volume center via $M_{\text{centered}}(t) = T(-\mathbf{c}) M(t) T(+\mathbf{c})$ (implemented in `scripts/compute_athena_fd.py`, commit `3c2b7ac`) eliminated this artifact and raised pooled translation correlation from $\rho = 0.8757$ to **$\rho = 0.9704$** (NeuroIMAGE: $0.5339 \to 0.8809$; Peking_1: $0.8022 \to 0.9901$).

---

## 3. Per-Site & Pooled Validation Results ($N = 404$ Development `rest_1` Scans)

All NIAK transformations were shifted to the scan's NIfTI volume center, trimmed by dropping the first 3 frames, and re-referenced to frame 0 of the retained run ($M_0(t) = M(0)^{-1} M(t)$) to match Athena's `3dvolreg -base rm_da[0]`.  
**Flag criteria:** Spearman $\rho < 0.80$ **OR** scale difference $> 25\%$ (median ratio $< 0.75$ or $> 1.25$).

### 3.1. Max Translation (mm): NIAK Volume-Centered $L_2$ vs. Athena `Max Motion (mm)`

| Site / Group | $N$ | Spearman $\rho$ | $p$-value | Median Ratio (NIAK/Athena) | Bland-Altman Bias [95% LoA] (mm) | Flag Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KKI** | 83 | 0.9625 | $1.18 \times 10^{-47}$ | 1.0644 | $+0.043$ $[-0.233, +0.319]$ | OK |
| **NeuroIMAGE** | 48 | 0.8809 | $1.50 \times 10^{-16}$ | 1.1372 | $+1.922$ $[-22.634, +26.478]$ | OK *(see Outlier Note)* |
| **OHSU** | 79 | 0.9784 | $1.76 \times 10^{-54}$ | 1.1099 | $-0.038$ $[-1.376, +1.300]$ | OK |
| **Peking_1** | 85 | 0.9901 | $1.65 \times 10^{-72}$ | 1.1636 | $+0.155$ $[+0.013, +0.298]$ | OK |
| **Peking_2** | 67 | 0.9743 | $8.60 \times 10^{-44}$ | 1.2120 | $+0.128$ $[-0.115, +0.370]$ | OK |
| **Peking_3** | 42 | 0.9910 | $1.46 \times 10^{-36}$ | 1.1283 | $+0.151$ $[-0.134, +0.437]$ | OK |
| *(Peking Combined)* | *(194)* | *(0.9858)* | *($1.06 \times 10^{-150}$)* | *(1.1636)* | *($+0.145$ $[-0.071, +0.361]$)* | *(OK)* |
| **POOLED** | **404** | **0.9704** | **$4.49 \times 10^{-250}$** | **1.1208** | **$+0.299$ $[-8.192, +8.790]$** | **OK** |

### 3.2. Max Rotation (deg): NIAK 3-Axis $L_2$ Norm vs. Athena `Max Rotation (degree)`

| Site / Group | $N$ | Spearman $\rho$ | $p$-value | Median Ratio (NIAK/Athena) | Bland-Altman Bias [95% LoA] (deg) | Flag Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KKI** | 83 | 0.9728 | $3.05 \times 10^{-53}$ | 1.1140 | $+0.087$ $[-0.451, +0.625]$ | OK |
| **NeuroIMAGE** | 48 | 0.9187 | $3.49 \times 10^{-20}$ | **1.2665** | $+2.187$ $[-22.997, +27.371]$ | **FLAG: `scale>25%`** |
| **OHSU** | 79 | 0.9845 | $6.18 \times 10^{-60}$ | **1.3283** | $+0.085$ $[-3.367, +3.538]$ | **FLAG: `scale>25%`** |
| **Peking_1** | 85 | 0.9680 | $1.36 \times 10^{-51}$ | 1.1598 | $+0.121$ $[-0.154, +0.396]$ | OK |
| **Peking_2** | 67 | 0.9796 | $4.77 \times 10^{-47}$ | **1.2577** | $+0.148$ $[-0.131, +0.428]$ | **FLAG: `scale>25%`** |
| **Peking_3** | 42 | 0.9556 | $7.93 \times 10^{-23}$ | **1.2994** | $+0.158$ $[-0.215, +0.532]$ | **FLAG: `scale>25%`** |
| *(Peking Combined)* | *(194)* | *(0.9720)* | *($9.32 \times 10^{-123}$)* | *(1.2153)* | *($+0.139$ $[-0.162, +0.439]$)* | *(OK)* |
| **POOLED** | **404** | **0.9730** | **$4.33 \times 10^{-258}$** | **1.2135** | **$+0.361$ $[-8.477, +9.199]$** | **OK** |

### 3.3. Max Rotation (deg): NIAK Single-Axis $L_\infty$ ($\max_{t,a}|\theta_{t,a}|$) vs. Athena `Max Rotation (degree)`

| Site / Group | $N$ | Spearman $\rho$ | $p$-value | Median Ratio (NIAK/Athena) | Bland-Altman Bias [95% LoA] (deg) | Flag Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KKI** | 83 | 0.9589 | $4.29 \times 10^{-46}$ | 1.0078 | $-0.006$ $[-0.602, +0.590]$ | OK |
| **NeuroIMAGE** | 48 | 0.9025 | $1.90 \times 10^{-18}$ | 1.0827 | $+1.803$ $[-21.135, +24.740]$ | OK |
| **OHSU** | 79 | 0.9847 | $3.08 \times 10^{-60}$ | 1.2102 | $+0.007$ $[-3.447, +3.461]$ | OK |
| **Peking_1** | 85 | 0.9641 | $1.50 \times 10^{-49}$ | 1.0652 | $+0.017$ $[-0.318, +0.352]$ | OK |
| **Peking_2** | 67 | 0.9729 | $4.58 \times 10^{-43}$ | 1.1350 | $+0.065$ $[-0.280, +0.410]$ | OK |
| **Peking_3** | 42 | 0.9331 | $2.29 \times 10^{-19}$ | 1.1450 | $+0.043$ $[-0.409, +0.496]$ | OK |
| *(Peking Combined)* | *(194)* | *(0.9628)* | *($4.96 \times 10^{-111}$)* | *(1.0889)* | *($+0.039$ $[-0.328, +0.406]$)* | *(OK)* |
| **POOLED** | **404** | **0.9653** | **$1.88 \times 10^{-236}$** | **1.0801** | **$+0.233$ $[-7.834, +8.301]$** | **OK** |

---

## 4. Flag Summary & Outlier Disclosure
1. **Zero Correlation Flags ($\rho < 0.80$):** All sites and pooled comparisons pass $\rho \ge 0.88$ for both translation and rotation.
2. **Scale Flags ($>25\%$):** Under the 3-axis $L_2$ rotation norm, four sub-sites exceed $+25\%$ (`NeuroIMAGE`: 1.2665, `OHSU`: 1.3283, `Peking_2`: 1.2577, `Peking_3`: 1.2994; pooled = 1.2135, OK). Under the single-axis $L_\infty$ rotation norm (closer to single-axis rotational extrema), **zero sites** exceed 25% (range $1.0078\text{--}1.2102$, pooled $= 1.0801$). The residual $+8\text{--}12\%$ positive scale tendency in NIAK vs. Athena reflects expected algorithmic differences between MINC `minctracc` (5 mm blurred cross-correlation) and AFNI `3dvolreg` (Fourier least-squares), reinforcing the importance of the co-primary **no-exclusion sensitivity analysis** locked in [`DEVIATIONS.md`](DEVIATIONS.md).
3. **NeuroIMAGE Outlier:** One NeuroIMAGE scan exhibits a `minctracc` registration divergence ($\max |\theta| = 85.98^\circ$ on 203/258 frames vs. Athena $7.34^\circ$), which drives the wide NeuroIMAGE and pooled Bland-Altman limits.
