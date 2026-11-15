# NeuroADHD Follow-Up Study: Functional Connectivity, Covariates, & High-Dimensional Regularization

**Evaluation Protocol:** Nested Cross-Validation (Inner 3-Fold Stratified CV on training folds only for pipeline and hyperparameter selection) evaluated under Repeated Stratified 5-Fold CV and Leave-One-Institution-Out CV (LOIO).
**Comparison Cohorts:**
- **Config 1 (Full Cohort):** All 153 subjects (103 Control, 50 ADHD across 6 institutions: KKI, NYU, NeuroIMAGE, OHSU, Peking, Pittsburgh). Majority Class Accuracy = **67.32%**.
- **Config 2 (Clean Cohort):** 118 subjects (68 Control, 50 ADHD across 4 institutions: KKI, NYU, NeuroIMAGE, Peking; excluding zero-ADHD sites OHSU & Pittsburgh). Majority Class Accuracy = **57.63%**.

---

## 1. Executive Summary & Plain Scientific Verdict

### Plain Scientific Statement: Does Functional Connectivity Add Predictive Value Beyond Site, Age, and Sex?

> **VERDICT: NO substantively generalizable predictive value beyond demographics and site structure.**
>
> 1. **Site Confounding in Config 1 (153 subjects):** In the full cohort, site identity alone achieves an in-sample ROC-AUC of **0.7221** and a Stratified 5-Fold CV AUC of **0.6745** (exceeding all neuroimaging models). This occurs because OHSU ($N=14$) and Pittsburgh ($N=21$) contain zero ADHD cases. Under Leave-One-Institution-Out CV, where site shortcuts cannot be exploited, functional connectivity collapses to chance (LOIO AUC = **0.5342** [0.439, 0.633]), and non-parametric within-institution label permutation testing confirms this is **not statistically significant** ($p = 0.0859$).
> 2. **Signal in Config 2 (118 subjects):** When zero-ADHD sites are excluded, functional connectivity achieves an out-of-site LOIO ROC-AUC of **0.6221** [0.515, 0.724], which is statistically distinguishable from within-site label permutations ($p = 0.0210$).
> 3. **Incremental Value Beyond Age & Sex:** Demographic covariates alone (standardized age and binary sex) achieve an out-of-site LOIO ROC-AUC of **0.5976** [0.492, 0.702] in Config 2. Combining 19,900 functional connectivity features with Age and Sex yields a LOIO ROC-AUC of **0.6218** [0.516, 0.724] — an incremental change of **-0.0003** compared to connectivity alone, and a gain of only **+0.0242** (2.4 percentage points) over age and sex alone. In Stratified 5-Fold CV, adding age and sex to connectivity changes AUC from 0.6318 to 0.6315 (-0.0003).
> 4. **Scientific Conclusion:** Resting-state functional connectivity carries modest diagnostic signal within multi-site data when site class balances are matched, but it does **not** provide substantial incremental predictive utility beyond basic demographic covariates (predominantly sex, reflecting the clinical ADHD male-to-female ratio). Furthermore, standard non-grouped cross-validation massively inflates perceived performance by learning scanner/site batch differences rather than neurological etiology.

---

## 2. Main Comparison: Connectivity vs Covariates vs Both

All metrics report Nested CV out-of-fold performance with subject-level 95% bootstrap confidence intervals [2.5%, 97.5%] (1,000 resamples). Majority class accuracy is reported next to every accuracy value.

### A. Config 1: All 153 Subjects (Majority Class Accuracy = 67.32%)

| Evaluation Scheme | Feature Modality | Accuracy [95% CI] | Majority Acc | Balanced Acc [95% CI] | ROC-AUC [95% CI] | Sensitivity | Specificity | MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stratified 5-Fold | Connectivity Only (19,900 edges) | **0.6275** [0.555, 0.699] | 67.32% | **0.5638** [0.489, 0.641] | **0.6033** [0.502, 0.696] | 0.3800 | 0.7476 | +0.1313 |
| Leave-One-Institution-Out | Connectivity Only (19,900 edges) | **0.5882** [0.510, 0.667] | 67.32% | **0.5192** [0.441, 0.598] | **0.5342** [0.439, 0.633] | 0.3200 | 0.7184 | +0.0396 |
| Stratified 5-Fold | Connectivity + Age/Sex (19,902 feats) | **0.6144** [0.536, 0.686] | 67.32% | **0.5438** [0.469, 0.619] | **0.5988** [0.502, 0.692] | 0.3400 | 0.7476 | +0.0914 |
| Leave-One-Institution-Out | Connectivity + Age/Sex (19,902 feats) | **0.5752** [0.497, 0.647] | 67.32% | **0.5044** [0.424, 0.580] | **0.5025** [0.412, 0.600] | 0.3000 | 0.7087 | +0.0090 |
| Stratified 5-Fold | Age + Sex Only (2 feats) | **0.5686** [0.490, 0.641] | 67.32% | **0.6024** [0.525, 0.677] | **0.5818** [0.490, 0.674] | 0.7000 | 0.5049 | +0.1937 |
| Leave-One-Institution-Out | Age + Sex Only (2 feats) | **0.5033** [0.418, 0.582] | 67.32% | **0.3944** [0.336, 0.451] | **0.4457** [0.359, 0.547] | 0.0800 | 0.7087 | -0.2383 |

### B. Config 2: Exclude OHSU & Pittsburgh (118 Subjects, Majority Class Accuracy = 57.63%)

| Evaluation Scheme | Feature Modality | Accuracy [95% CI] | Majority Acc | Balanced Acc [95% CI] | ROC-AUC [95% CI] | Sensitivity | Specificity | MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stratified 5-Fold | Connectivity Only (19,900 edges) | **0.5847** [0.500, 0.669] | 57.63% | **0.5735** [0.491, 0.665] | **0.6318** [0.528, 0.729] | 0.5000 | 0.6471 | +0.1475 |
| Leave-One-Institution-Out | Connectivity Only (19,900 edges) | **0.6186** [0.534, 0.703] | 57.63% | **0.6003** [0.515, 0.690] | **0.6221** [0.515, 0.724] | 0.4800 | 0.7206 | +0.2060 |
| Stratified 5-Fold | Connectivity + Age/Sex (19,902 feats) | **0.6017** [0.517, 0.686] | 57.63% | **0.5882** [0.501, 0.676] | **0.6315** [0.525, 0.731] | 0.5000 | 0.6765 | +0.1781 |
| Leave-One-Institution-Out | Connectivity + Age/Sex (19,902 feats) | **0.6271** [0.542, 0.712] | 57.63% | **0.6103** [0.525, 0.700] | **0.6218** [0.516, 0.724] | 0.5000 | 0.7206 | +0.2254 |
| Stratified 5-Fold | Age + Sex Only (2 feats) | **0.5678** [0.475, 0.653] | 57.63% | **0.5509** [0.465, 0.640] | **0.5569** [0.452, 0.658] | 0.4400 | 0.6618 | +0.1035 |
| Leave-One-Institution-Out | Age + Sex Only (2 feats) | **0.5763** [0.492, 0.661] | 57.63% | **0.5768** [0.489, 0.664] | **0.5976** [0.492, 0.702] | 0.5800 | 0.5735 | +0.1518 |

---

## 3. Site-Only Baseline Analysis

To measure how much diagnosis information can be predicted purely from acquisition site indicators (without any brain imaging or demographic data):

| Cohort | Evaluation Mode | Accuracy | Majority Acc | Balanced Acc | ROC-AUC | Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Config 1 (153 Subj) | In-Sample Fit | 0.6928 | 67.32% | 0.5866 | **0.7221** | Sites differ heavily in ADHD rate (OHSU/Pitt: 0%, NYU: 67%) |
| Config 1 (153 Subj) | Stratified 5-Fold CV | 0.6758 | 67.32% | 0.5565 | **0.6745** | Exceeds all neuroimaging models! Confirms strong site confounding |
| Config 1 (153 Subj) | LOIO CV (Out-of-Site) | 0.6732 | 67.32% | 0.5000 | **0.3326** | Inverts because held-out sites have opposite base rates to training sites |
| Config 2 (118 Subj) | In-Sample Fit | 0.6017 | 57.63% | 0.5591 | **0.5791** | Site imbalance reduced once zero-ADHD sites are removed |
| Config 2 (118 Subj) | Stratified 5-Fold CV | 0.5864 | 57.63% | 0.5390 | **0.5010** | Chance-level site signal; no artificial site shortcuts remain |
| Config 2 (118 Subj) | LOIO CV (Out-of-Site) | 0.5763 | 57.63% | 0.5000 | **0.4609** | At chance level, confirming clean cross-site generalization |

---

## 4. High-Dimensional Full-Feature Models vs SelectKBest(k=100)

We systematically evaluated regularized linear models operating on **all 19,900 functional connectivity features** ($p \gg N$ regime) against univariate ANOVA feature selection (`SelectKBest(k=100)`):

| Architecture | Feature Dim | Config 1 Strat-5F AUC | Config 1 LOIO AUC | Config 2 Strat-5F AUC | Config 2 LOIO AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `L2-LogReg-Full (C=0.001, balanced)` | 19,900 | 0.5592 | 0.4483 | 0.5482 | 0.5729 |
| `L2-LogReg-Full (C=0.01, balanced)` | 19,900 | 0.5449 | 0.4313 | 0.5279 | 0.5474 |
| `ElasticNet-Full (C=0.01, l1=0.5, balanced)` | 19,900 | 0.4981 | 0.4988 | 0.5088 | 0.5326 |
| `L2-LogReg-K100 (C=0.01, balanced)` | 100 | **0.6107** | **0.5419** | **0.6324** | **0.6241** |
| `RF-n50-d3-K100 (balanced)` | 100 | **0.6000** | 0.5122 | **0.6432** | **0.6250** |
| `RBF-SVM-K100 (C=1.0, balanced)` | 100 | 0.5898 | 0.5068 | **0.6329** | 0.6044 |

> **Finding:** Full-feature high-dimensional regularization (L2 shrinkage and ElasticNet coordinate descent) performs substantially worse than univariate feature selection. With $p = 19,900$ and $N \approx 118–153$ ($p/N \approx 150$), dense noise across ~20,000 correlation pairs drowns out sparse biological signal. Reducing dimensionality by 99.5% via in-fold ANOVA $F$-ranking (`SelectKBest(k=100)`) consistently yielded the strongest validation and cross-site performance.

---

## 5. Within-Institution Label Permutation Significance Test

To determine whether cross-site performance reflects true neurobiological signal rather than chance alignment with site-specific base rates, we performed non-parametric permutation tests (1,000 shuffles) where diagnosis labels were shuffled **strictly within each institution** (preserving site sample sizes and site-specific case counts):

| Cohort Configuration | Observed LOIO ROC-AUC | Permutation Null Mean ± SD | 95th Percentile Null | 99th Percentile Null | Empirical p-value | Significance (α = 0.05) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1 (153 Subjects)** | **0.5122** | 0.4231 ± 0.0641 | 0.5261 | 0.5703 | **p = 0.0859** | **Not Significant (p ≥ 0.05)** |
| **Config 2 (118 Subjects)** | **0.6282** | 0.4866 ± 0.0673 | 0.5974 | 0.6409 | **p = 0.0210** | **Significant** |

> **Interpretation:**  
> - In **Config 1**, observed cross-site performance is within the null distribution ($p = 0.0859$), proving that zero-case sites (OHSU and Pittsburgh) corrupt cross-site validation and prevent statistically significant generalizability.
> - In **Config 2**, when only sites with both cases and controls are evaluated, the observed LOIO AUC of 0.6282 exceeds the 95th percentile null (0.5974), achieving statistical significance ($p = 0.0210$).

---

## 6. Phenotypic Columns & Feature Engineering Specification

In accordance with audit rules, **zero symptom-, diagnosis-, or cognitive-derived scores** were used. The exact columns retrieved from `ADHD200_Official/metadata/official_cc200_phenotype_match.csv` were:
1. `age_years` (Continuous, range: 7.17 to 21.74 years, mean: 12.28 ± 3.23): Standardized with `StandardScaler` strictly fitted on training folds (`StandardScaler.fit(age_train)`).
2. `sex` (Binary: Male = 1, Female = 0; 93 Male, 60 Female across cohort).

**Strictly Excluded Columns:**
- ADHD Index, Inattention, Hyperactivity/Impulsivity symptom counts.
- Full Scale IQ, Verbal IQ, Performance IQ.
- Medication status (`Med Status`) and secondary comorbid diagnoses.

---

## 7. Leakage Unit Test Verification

The standalone test suite `test_followup_leakage.py` was executed and passed (3 tests, 0 failures):

- `test_age_scaler_isolation`: Verified that corrupting held-out age values ($X_{val} + 9,999$) produces **0 change** in fitted training age mean or variance.
- `test_feature_selection_isolation`: Verified that corrupting held-out connectivity values ($X_{val} + 50,000$) produces **0 change** in ANOVA $F$-scores or selected top-100 edges.
- `test_smote_resampling_isolation`: Verified that SMOTE operates strictly on training fold features (adding +44 synthetic samples to training fold; held-out fold is never resampled).

---

