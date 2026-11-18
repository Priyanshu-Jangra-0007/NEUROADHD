# Presentation Outline: NeuroADHD Benchmark & Methodology Defense

This document provides a slide-by-slide presentation outline for a technical defense, viva examination, or peer review of the **NeuroADHD** resting-state fMRI CC200 machine learning benchmark.

---

### Slide 1: Problem & Motivation
* **Title**: Machine Learning on Resting-State fMRI for ADHD: Promise vs. Reality
* **Key Points**:
  - Attention-Deficit/Hyperactivity Disorder (ADHD) is traditionally diagnosed through clinical observation and subjective behavioral rating scales; neuroimaging biomarkers remain an elusive research goal.
  - Resting-state functional magnetic resonance imaging (rs-fMRI) captures spontaneous blood-oxygen-level-dependent (BOLD) low-frequency fluctuations, from which whole-brain functional connectivity networks are derived.
  - Neuroimaging machine learning faces notorious failure modes: the "curse of dimensionality" ($p \gg N$), site-specific scanner artifacts, publication bias toward over-optimistic random cross-validation, and non-replicable clinical claims.
  - *Goal of this project*: Build a rigorous, fully auditable, leak-free benchmark on the ADHD-200 CC200 dataset to honestly determine whether resting-state functional connectivity predicts ADHD beyond demographic covariates and scanner confounds.
* **Source Reference**: [README.md](file:///Users/apple/Documents/Codex/2026-09-29/down/README.md)

---

### Slide 2: The Dataset & Feature Space
* **Title**: Dataset Architecture & High-Dimensional Connectivity Space
* **Key Points**:
  - Utilizes the public ADHD-200 Consortium sample preprocessed by the Neuro Bureau via C-PAC using the Craddock CC200 functional parcellation atlas ($200$ regions of interest).
  - Full cohort comprises $N = 153$ subjects across 6 institutions: 103 typically developing controls and 50 individuals diagnosed with ADHD.
  - Each subject contributes an ROI time-series CSV; functional connectivity is calculated via pairwise Pearson correlation followed by Fisher $z$-transformation:
    $$\text{Feature Dimension} = \frac{200 \times 199}{2} = 19,900 \text{ unique functional edges}$$
  - Presents a massive $p \gg N$ ratio ($19,900$ features for $118\text{--}153$ subjects), necessitating strict feature selection and leakage prevention.
* **Source Reference**: [NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv), [benchmarkData.js](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelDashboard/frontend/src/data/benchmarkData.js)

---

### Slide 3: Pipeline Architecture & Leakage Prevention
* **Title**: Leak-Free Preprocessing & Gated Model Architecture
* **Key Points**:
  - Implements a scikit-learn pipeline strictly fit *only* on training data within each fold:
    $$\text{Median Imputation} \longrightarrow \text{Variance Threshold (0.0)} \longrightarrow \text{StandardScaler} \longrightarrow \text{SelectKBest}(F\text{-score}, k=100)$$
  - Model architecture: Shallow Balanced Random Forest (`n_estimators=50`, `max_depth=3`, `class_weight='balanced'`, `max_features='sqrt'`) to resist overfitting on small sample sizes.
  - Frozen model checkpoint: `final_model.joblib` (74,409 bytes) and `final_preprocessing_pipeline.joblib`.
  - Double-gated test evaluation in CLI: requires both `--evaluate-test` and `--confirm-test-eval` flags to prevent accidental leakage into the held-out test split.
* **Source Reference**: [NeuroADHD_ModelBenchmark/results/final_locked/final_model_metadata.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/results/final_locked/final_model_metadata.json), [NeuroADHD_ModelBenchmark/scripts/run_all_models.py](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/scripts/run_all_models.py)

---

### Slide 4: Validation Strategy: Why Random Folds Lie
* **Title**: Why random folds can inflate results
* **Key Points**:
  - Standard random $k$-fold cross-validation mixes subjects from the same scanning site across training and validation folds.
  - When scanner sites differ in baseline noise, coil characteristics, or demographic composition, classifiers inadvertently learn to classify *scanner site* rather than ADHD pathology.
  - Two distinct evaluation protocols deployed:
    1. *Peking_2 Held-Out Partition* ($N=20$): Fixed institution holdout, but shares the same 3T Siemens Trio scanner as Peking_1 and Peking_3 training partitions (same-scanner batch reproducibility).
    2. *Leave-One-Institution-Out (LOIO) Cross-Validation*: True out-of-site generalization, rotating each entire clinical institution out as the test fold.
* **Source Reference**: [NeuroADHD/configs/splits/cc200_site_holdout.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD/configs/splits/cc200_site_holdout.json), [siteConfoundData.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelDashboard/frontend/src/data/siteConfoundData.json)

---

### Slide 5: Primary Locked Model Results
* **Title**: Benchmark Findings: Peking_2 Held-Out Evaluation ($N=20$)
* **Key Points**:
  - Peking_2 held-out test metrics ($N=20$, 10 ADHD, 10 Control):
    - Accuracy: **`60.00%`** (12/20 correct), Wilson 95% CI: **`[38.7%, 78.1%]`**.
    - ROC-AUC: **`0.5600`**, Stratified bootstrap 95% CI: **`[0.290, 0.830]`** (2,000 resamples).
    - PR-AUC: **`0.6341`** (vs chance `0.5000`).
    - Confusion Matrix: TP = 6, TN = 6, FP = 4, FN = 4 (Balanced Acc: 60.0%, Sensitivity: 60.0%, Specificity: 60.0%).
  - Evaluated strictly once for this locked Random Forest checkpoint; however, an early baseline model had previously been scored on it during preliminary exploratory work.
  - Wide confidence intervals span chance performance ($50\%$), underscoring the statistical uncertainty inherent in small holdout test sets ($N=20$).
* **Source Reference**: [NeuroADHD_ModelBenchmark/results/final_locked/final_test_results.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/results/final_locked/final_test_results.json), [final_confusion_matrix.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/results/final_locked/final_confusion_matrix.json)

---

### Slide 6: The Multi-Site Confound Finding
* **Title**: Scanner Confounding & Leave-One-Institution-Out (LOIO) Reality
* **Key Points**:
  - In the full $N=153$ cohort, an in-sample logistic regression using *only one-hot site indicators* reaches ROC-AUC **`0.7221`**, demonstrating severe site-diagnosis association.
  - Root cause: Two institutions contributed zero ADHD cases to the CC200 release:
    - **OHSU**: 14 subjects, 0 ADHD
    - **Pittsburgh**: 21 subjects, 0 ADHD
  - Full cohort ($N=153$) connectivity LOIO performance collapses to ROC-AUC **`0.5342`** [0.439, 0.633] and Accuracy **`58.82%`** [0.510, 0.667] (permutation $p = 0.0859$, not statistically significant).
  - Clean cohort ($N=118$, removing OHSU & Pittsburgh, merging Peking 1/2/3): Connectivity LOIO ROC-AUC is **`0.6221`** [0.515, 0.724], Accuracy **`61.86%`**, permutation $p = 0.0210$ *(exploratory post-hoc subset; did not hold up in the Study 2 pre-registered follow-up on a different preprocessing pipeline)*.
* **Source Reference**: [NeuroADHD_ModelDashboard/frontend/src/data/siteConfoundData.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelDashboard/frontend/src/data/siteConfoundData.json)

---

### Slide 7: Demographic Covariates vs. Functional Connectivity
* **Title**: Do Connectivity Features Beat Simple Demographics?
* **Key Points**:
  - A baseline model trained *solely* on standardized age and biological sex reaches LOIO ROC-AUC **`0.5857`** [0.489, 0.685] (Accuracy `61.02%`, Balanced Accuracy `63.26%`, permutation $p = 0.0498$).
  - Apples-to-apples comparison on the 118-subject clean cohort *(Study 1 exploratory — did not hold up in the Study 2 follow-up)*:
    - Locked Random Forest (Connectivity): LOIO AUC **`0.6250`** [0.518, 0.726], Accuracy `61.86%` *(exploratory; did not hold up in Study 2)*.
    - Paired difference: $\Delta \text{AUC} = +0.0397$ [$-0.1061, +0.1695$], paired permutation **`p = 0.2670`** (not statistically significant).
    - Nested Ensemble (126 configs): LOIO AUC **`0.6253`** [0.521, 0.728], $\Delta \text{AUC} = +0.0400$, paired permutation **`p = 0.2685`** (not significant).
  - *Core scientific takeaway*: Functional connectivity did **not** demonstrate a statistically significant performance advantage over age and sex alone. The demographic signal reflects cohort recruitment differences, not ADHD-specific brain circuitry.
* **Source Reference**: [NeuroADHD_ModelBenchmark/experiments/model_comparison/model_comparison_results.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/experiments/model_comparison/model_comparison_results.json), [modelComparisonData.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelDashboard/frontend/src/data/modelComparisonData.json)

---

### Slide 8: What Was Tried That Did Not Help
* **Title**: Rigorous Negative Findings & Ablation Studies
* **Key Points**:
  - **Class Balancing**: SMOTE and RandomUnderSampler led to decision boundary distortion and failed to beat scikit-learn's native `class_weight='balanced'` under LOIO.
  - **Model Families**: Linear SVM, RBF SVM, ExtraTrees, and XGBoost suffered severe variance and failed to outperform a simple depth-3 Random Forest.
  - **Precision & Covariance Representations**: Ledoit-Wolf shrinkage partial correlation and Riemannian tangent space projections yielded negligible changes over flat Fisher $z$-Pearson matrices.
  - **126-Configuration Nested CV Search**: Exhaustive inner-loop optimization across correlation types, feature cutoffs, and classifiers converged to simple trees and showed zero statistical gain over the locked RF ($\Delta \text{AUC} = +0.0003, p = 0.4960$).
* **Source Reference**: [experiments/balancing_study/REPORT.md](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/experiments/balancing_study/REPORT.md), [experiments/final_candidate/REPORT.md](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/experiments/final_candidate/REPORT.md)

---

### Slide 9: Study 2 — Pre-Registered Follow-Up on a Different Preprocessing Pipeline (Athena CC200)
* **Title**: Pre-Registered Follow-Up on Athena CC200: Section 8.1 Not Met; Sealed NYU Inconclusive
* **Key Points**:
  - Protocol preregistered on OSF prior to Athena data acquisition ([https://osf.io/grbt2](https://osf.io/grbt2) and [https://osf.io/db7nv](https://osf.io/db7nv), commit `2d7ed3b`) locking the exact Study 1 classifier pipeline (`SelectKBest(k=100)` + `RandomForestClassifier(50, max_depth=3, balanced, random_state=42)`) across four Athena development institutions (KKI, NeuroIMAGE, OHSU, Peking) and a single one-shot evaluation on sealed NYU. Study 2 differs from Study 1 in preprocessing pipeline (Athena vs. C-PAC), atlas (`190` populated ROIs vs. `200`), cohort, and sites (with `126` development subjects and `6` NYU subjects overlapping Study 1).
  - **Development Primary LOIO ($N = 368$, Section 5.3 FD exclusion):** fMRI ROC-AUC **`0.5241`** `[0.4691, 0.5797]`, $p_{\text{perm}} = 0.2298$; Motion baseline `0.6038`, Age/Sex baseline `0.6163` ($\Delta\text{AUC}_{\text{motion}} = -0.0797$ `[-0.1520, -0.0099]`). **Section 8.1 NOT MET.**
  - **Development Registered Section 5.4 Sensitivity LOIO ($N = 404$, no exclusion):** fMRI ROC-AUC **`0.5740`** `[0.5243, 0.6243]`, $p_{\text{perm}} = 0.0050$; Motion baseline `0.6158`, Age/Sex baseline `0.5857` ($\Delta\text{AUC}_{\text{motion}} = -0.0418$ `[-0.0890, +0.0087]`). **Section 8.1 NOT MET.**
  - **One-Shot Sealed NYU External Evaluation:** Primary ($N = 196$, governing): **`0.5765`** `[0.496177, 0.656518]` (descriptive Motion `0.6069`, Age/Sex `0.5806`); Sensitivity ($N = 216$): **`0.5770`** `[0.499996, 0.652819]` (descriptive Motion `0.6076`, Age/Sex `0.6150`). Both fall in the registered **Inconclusive Zone** (confidence interval spans chance; external validity cannot be claimed).
  - **Takeaways:** Study 1's exploratory LOIO AUC (`~0.62`) did not hold up in this pre-registered follow-up (not a direct test of the identical C-PAC protocol); motion-only and age/sex baselines scored as high or higher than fMRI in all four regimes. This conclusion applies to this locked pipeline, not to connectivity in general. **The sealed NYU set is used up; any further modeling is exploratory.**
* **Source Reference**: `docs/study2/STUDY2_REPORT.md`, `docs/study2/STUDY2_SUMMARY.md`, `NeuroADHD_ModelDashboard/frontend/src/data/study2Data.json`

---

### Slide 10: Limitations & Ethical Guardrails
* **Title**: Methodological Honesty & Study Boundaries
* **Key Points**:
  - *Post-Hoc Study 1 Cohort Selection:* The 118-subject clean cohort was selected post-hoc after the full 153-subject cohort showed no statistically significant connectivity signal ($p = 0.0859$), and its `~0.62` LOIO AUC did not hold up in Study 2 (a pre-registered follow-up differing in preprocessing pipeline, atlas [`190` ROIs], cohort, and sites).
  - *Sealed Set Used Up & Prior Exposure:* The sealed NYU set has been scored once and is used up; any further modeling is exploratory. Moreover, `126` Study 2 development subjects and `6` NYU subjects overlapped with Study 1's C-PAC cohort where hyperparameters were chosen (the OSF foreknowledge option understates this), and only the OSF registrations are externally timestamped (the freeze/results tags were pushed to `origin` after the sealed run).
  - *Scope of Conclusion & Power:* $N=20$ for Peking_2 (Study 1) and $N=196/216$ for NYU (Study 2, CI half-width $\approx \pm 0.08$) limit statistical precision; wide intervals do not rule out a small population effect, and the conclusion applies to this locked pipeline rather than to functional connectivity in general.
  - *Acquisition Protocol Heterogeneity & Motion:* TR, TE, voxel size, and head motion vary widely across centers; motion-only and age/sex baselines scored as high or higher than fMRI in Study 2.
  - *Strict Ethical Disclaimer:* **Research use only: not a medical device or diagnostic tool; model scores are not calibrated probabilities.** Must never be used for clinical psychiatric diagnosis, triage, or treatment planning.
* **Source Reference**: `NeuroADHD_ModelDashboard/frontend/src/views/LimitationsView.jsx`, `docs/study2/STUDY2_REPORT.md`

---

### Slide 11: Interactive Exploration Dashboard
* **Title**: NeuroADHD Dashboard: Transparent Visual Analytics
* **Key Points**:
  - Full-stack React + Vite frontend and FastAPI backend built for reproducibility and open-science auditability.
  - 10 comprehensive views:
    - *Overview & Model Pipeline*: Cohort breakdowns, architecture specifications, locked hyperparameters.
    - *Single-Subject Inference & Decompositions*: Local TreeSHAP waterfall decompositions (anonymized Cases A–E).
    - *Performance & Hybrids*: Repetition logs, Peking_2 holdout, model comparison tables.
    - *Explainability & Connectivity Network*: Global SHAP edge importances and topological graph visualization of top predictive hubs.
    - *Methodology & Limitations*: Detailed split documentation, site confound analyses, and ethical constraints.
  - Zero subject IDs displayed; fully responsive across desktop (1440px) and mobile (390px) with full keyboard accessibility.
* **Source Reference**: [NeuroADHD_ModelDashboard/frontend/src/App.jsx](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelDashboard/frontend/src/App.jsx)

---

### Slide 12: Summary & Next Steps
* **Title**: Key Takeaways & Recommendations for Future Neuroimaging ML
* **Key Points**:
  - Publishing negative and inconclusive preregistered results is essential to counter publication bias in computational psychiatry and neuroimaging.
  - Without Leave-One-Institution-Out cross-validation, head-motion controls, and demographic baselines, machine learning models frequently report inflated performance reflecting non-neural artifacts.
  - In Study 2 (a pre-registered follow-up on a different preprocessing pipeline), Section 8.1 criteria were not met and sealed NYU fell in the Inconclusive Zone; this conclusion applies to this locked pipeline rather than connectivity in general, and **the sealed NYU set is used up so any further modeling on ADHD-200 is exploratory.**
  - *Future directions & next steps (exploratory or on independent new cohorts)*:
    - **Scanner Harmonization (ComBat / neuroCombat)**: Empirically evaluating empirical Bayes harmonization on multi-site cohorts with balanced site distributions (untested in this benchmark due to out-of-site CV constraints).
    - **Large Prospective Cohorts**: Standardized multi-site acquisition protocols with large samples (e.g., ABCD Study, $N > 10,000$).
    - **Multimodal Integration**: Fusing resting-state functional connectivity with T1w structural volumetrics, DTI tractography, and cognitive batteries.
* **Source Reference**: `README.md`, `docs/study2/STUDY2_REPORT.md`

---

### Slide 13: Likely Viva Questions & Honest Defenses
* **Title**: Viva Defense Q&A: Addressing Critical Methodological Inquiries

1. **Q: Was the held-out Peking_2 test set really untouched?**
   - **A**: The final model was scored once on Peking_2; the Peking_2 set was scored 10 times in total (9 baseline runs, 1 final); no tuning used it; same institution and scanner as Peking_1/3. Therefore, our primary claims of generalizability rest on the Leave-One-Institution-Out (LOIO) cross-validation (where Peking_1, Peking_2, and Peking_3 are unified into a single institution fold), not on Peking_2.
   - *Reference*: [NeuroADHD_ModelBenchmark/results/final_locked/final_test_results.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/results/final_locked/final_test_results.json)

2. **Q: Why did you exclude OHSU and Pittsburgh from the 118-subject clean cohort?**
   - **A**: OHSU ($N=14$) and Pittsburgh ($N=21$) contain zero ADHD cases in the CC200 preprocessed distribution (100% control). When left in LOIO cross-validation, any test fold on these sites tests a classifier with undefined specificity or sensitivity for ADHD, distorting macro-averaged metrics. However, we explicitly disclose both the 153-subject and 118-subject results and clarify that the 118-subject cohort was an exploratory post-hoc partition.
   - *Reference*: [NeuroADHD_ModelDashboard/frontend/src/data/siteConfoundData.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelDashboard/frontend/src/data/siteConfoundData.json)

3. **Q: Is this model clinically useful for diagnosing ADHD?**
   - **A**: *No.* Study 1's exploratory LOIO AUC (`0.6250` on $N=118$) did not beat age and sex alone ($p = 0.2670$) and did not hold up in Study 2 — a pre-registered follow-up on a different preprocessing pipeline (Athena, `190` ROIs, different cohort/sites with `126` overlapping development subjects; primary $N = 368$ LOIO AUC `0.5241`, $p_{\text{perm}} = 0.2298$; sealed NYU primary $N = 196$ AUC `0.5765` `[0.496177, 0.656518]`, Inconclusive Zone), where motion-only and age/sex baselines scored as high or higher than fMRI in every regime. This conclusion applies to this pipeline rather than to connectivity in general. Research use only: not a medical device or diagnostic tool; model scores are not calibrated probabilities.
   - *Reference*: [NeuroADHD_ModelBenchmark/experiments/model_comparison/model_comparison_results.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/experiments/model_comparison/model_comparison_results.json)

4. **Q: Why wasn't ComBat evaluated on this data, and why did Tangent Space not help?**
   - **A**: ComBat estimates location and scale parameters per site. In strict Leave-One-Institution-Out CV, the held-out institution is completely unseen at training time, so standard training-set ComBat cannot estimate parameters for the held-out site without test-site data (verified via unit test in `test_candidate_leakage.py`). Riemannian tangent space was evaluated across the 118 subjects, but noisy empirical covariance estimates on short time series ($T \approx 120\text{--}240$) prevented it from improving over Fisher $z$-Pearson connectivity.
   - *Reference*: [NeuroADHD_ModelBenchmark/experiments/final_candidate/test_candidate_leakage.py](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/experiments/final_candidate/test_candidate_leakage.py), [experiments/final_candidate/REPORT.md](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/experiments/final_candidate/REPORT.md)
