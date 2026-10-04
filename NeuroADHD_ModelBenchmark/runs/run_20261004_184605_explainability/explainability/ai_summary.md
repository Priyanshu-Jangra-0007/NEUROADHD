# NeuroADHD Model Explainability & Feature Interpretation Report

**Run ID:** `run_20261004_184605_explainability`  
**Target Model:** Frozen Balanced Random Forest (K=100, n_estimators=50, max_depth=3, class_weight="balanced", seed=42)  
**Development Cohort:** 133 Subjects (93 Control, 40 ADHD across 7 sites)  
**Peking_2 Test Cohort:** 20 Subjects — **INTERPRETATION ONLY (STRICTLY LOCKED)**  
**Parcellation Atlas:** Craddock 200 (CC200) Functional Clustering Atlas  

---

## 1. Top 10 Predictive Functional Connectivity Edges

| Rank | Connectivity Edge | ROI Pair | Mean \|SHAP\| | Associated Model Direction |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `ROI_023__ROI_189` | ROI_023 — ROI_189 | `0.0168` | Higher connectivity increases ADHD probability |
| 2 | `ROI_100__ROI_153` | ROI_100 — ROI_153 | `0.0162` | Higher connectivity increases ADHD probability |
| 3 | `ROI_034__ROI_041` | ROI_034 — ROI_041 | `0.0155` | Higher connectivity increases ADHD probability |
| 4 | `ROI_143__ROI_153` | ROI_143 — ROI_153 | `0.0143` | Higher connectivity increases ADHD probability |
| 5 | `ROI_023__ROI_103` | ROI_023 — ROI_103 | `0.0141` | Higher connectivity increases ADHD probability |
| 6 | `ROI_003__ROI_006` | ROI_003 — ROI_006 | `0.0139` | Higher connectivity increases ADHD probability |
| 7 | `ROI_056__ROI_138` | ROI_056 — ROI_138 | `0.0132` | Higher connectivity increases ADHD probability |
| 8 | `ROI_026__ROI_153` | ROI_026 — ROI_153 | `0.0127` | Higher connectivity increases ADHD probability |
| 9 | `ROI_034__ROI_103` | ROI_034 — ROI_103 | `0.0126` | Higher connectivity increases ADHD probability |
| 10 | `ROI_006__ROI_127` | ROI_006 — ROI_127 | `0.0122` | Higher connectivity increases ADHD probability |

---

## 2. Top Hub ROIs Across Predictive Edges

| ROI Identifier | CC200 Cluster ID | Top-50 Edge Count | All-100 Edge Count |
| :--- | :---: | :---: | :---: |
| **ROI_003** | 3 | 8 | 16 |
| **ROI_034** | 34 | 6 | 9 |
| **ROI_006** | 6 | 4 | 6 |
| **ROI_153** | 153 | 3 | 4 |
| **ROI_189** | 189 | 6 | 7 |
| **ROI_151** | 151 | 4 | 5 |
| **ROI_023** | 23 | 3 | 6 |
| **ROI_100** | 100 | 3 | 4 |
| **ROI_103** | 103 | 2 | 2 |
| **ROI_174** | 174 | 6 | 9 |

---

## 3. Concordance: SHAP vs Permutation Importance

- **Top 10 Feature Overlap:** `7/10` (70%)
- **Top 20 Feature Overlap:** `14/20` (70%)
- **Top 50 Feature Overlap:** `40/50` (80%)

Both SHAP and Permutation importance demonstrate consistent consensus on the dominant functional connectivity subnetworks driving ADHD classification.

---

## 4. Peking_2 Locked Test Set Interpretation

- **Cohort:** 20 subjects (10 ADHD, 10 Control).
- **Test Integrity:** Model parameters, thresholds, and feature sets remain frozen. SHAP decompositions were generated purely for post-hoc validation interpretation.
- **Observations:** High concordance in directional feature impact between development cohort and held-out test cohort.
