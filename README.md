# NeuroADHD: Resting-State fMRI CC200 Benchmark & Research Dashboard

An academic machine learning benchmark and exploration dashboard investigating functional connectivity patterns in Attention-Deficit/Hyperactivity Disorder (ADHD) using the ADHD-200 CC200 resting-state fMRI dataset.

> [!WARNING]
> Research use only: not a medical device or diagnostic tool; model scores are not calibrated probabilities.

**Study 2 (Preregistered Multi-Site Athena Evaluation):** OSF Preregistration ([https://osf.io/grbt2](https://osf.io/grbt2)) and final summary ([`docs/study2/STUDY2_SUMMARY.md`](docs/study2/STUDY2_SUMMARY.md)).  
Study 2 is aggregate-only; OSF registrations: https://osf.io/grbt2 and https://osf.io/db7nv.

---

## 📁 Repository Architecture

The project is structured into three primary packages alongside centralized documentation and archives:

```text
.
├── NeuroADHD/                     # Core ML package & feature extraction
│   ├── configs/splits/            # Site holdout split definitions (cc200_site_holdout.json)
│   ├── data/adhd200/              # Dataset root (manifest, CC200 raw BOLD, features, splits)
│   ├── src/                       # Extraction, preprocessing & single-subject inference
│   │   ├── features/              # Pairwise Fisher z-transformed Pearson connectivity
│   │   ├── preprocessing/         # In-fold median imputation, scaling, & ANOVA selection
│   │   └── inference/             # predict_subject.py (CLI inference engine)
│   └── tests/                     # Preprocessing leakage & inference unit tests
│
├── NeuroADHD_ModelBenchmark/      # Benchmarking, LOIO validation, and experiment suites
│   ├── configs/                   # Benchmark configuration and per-model hyperparameter specs
│   ├── experiments/               # Balancing study, demographic covariates, and candidate studies
│   ├── results/final_locked/      # Frozen Balanced Random Forest checkpoint (74,409 bytes)
│   ├── runs/                      # 26 tracked benchmark runs with metrics and confusion matrices
│   ├── scripts/                   # Model training, evaluation, and report generators
│   └── tests/                     # Test-set locking unit tests (enforcing double-gated eval)
│
├── NeuroADHD_ModelDashboard/      # Visualization & interactive exploration dashboard
│   ├── dashboard.html             # Local standalone HTML preview (gitignored, preserved on disk)
│   ├── backend/                   # FastAPI server (:8000) for runs, metrics, and inference
│   │   ├── compile_dashboard_data.py # Portable script compiling frontend data
│   │   ├── server.py              # Sole backend application entry point
│   │   └── services/              # Benchmark data aggregation service
│   └── frontend/                  # React + Vite application (:5173)
│       ├── index.html             # Active Vite dev server entry point
│       └── src/views/             # Core views (Overview, Model, Methodology, Limitations)
│
├── archive/manifests/             # Historical backup manifests, cleanup logs, and inventories
├── scripts/ad_hoc/                # Ad-hoc utilities (downloaders, PDF generation)
└── ADHD200_Official/              # ADHD-200 official S3 manifests, audits, and metadata
```

---

## 🚀 Quick Start: Running the Project

### 1. Run the Backend API Server
The backend is a FastAPI application providing REST endpoints for benchmark leaderboards, experiment histories, and single-subject inference.

From the repository root:
```bash
./NeuroADHD/.venv/bin/python -m uvicorn server:app \
  --app-dir NeuroADHD_ModelDashboard/backend \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```
* **API Address**: `http://127.0.0.1:8000`
* **Interactive API Documentation (Swagger)**: `http://127.0.0.1:8000/docs`
* **Health Check**: `curl http://127.0.0.1:8000/api/health`

### 2. Run the Frontend Dashboard
The frontend is powered by Vite and pre-configured to proxy `/api` calls directly to the FastAPI backend.

In a separate terminal window:
```bash
cd NeuroADHD_ModelDashboard/frontend
npm run dev
```
* **Dev Server**: `http://localhost:5173`
* **Build for Production**: `npm run build`

### 3. Standalone Dashboard (Optional Local Preview)
The primary dashboard is the React Vite app (`http://localhost:5173`). An optional standalone static preview is preserved locally on disk (gitignored):
```bash
open NeuroADHD_ModelDashboard/dashboard.html
```

---

## 📊 Dataset Placement & Feature Extraction

### Data Placement
Raw Craddock CC200 ROI average time-series CSVs must be organized by subject directory under:
```text
NeuroADHD/data/adhd200/raw/cc200/<subject_id>_session_1/roi_CC200.csv
```
Each file must contain $T$ timepoint rows and 200 column headers representing mean BOLD signals per ROI.

### Rebuilding Features
To compute the $19,900$ unique pairwise functional connectivity edges ($200 \times 199 / 2$) using the Fisher $z$-transformed Pearson correlation matrix:

```bash
./NeuroADHD/.venv/bin/python NeuroADHD/src/features/extract_cc200_connectivity.py
```
This generates the full feature matrix at:
`NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv` (153 subjects: 103 Control, 50 ADHD).

---

## 🔒 Locked Final Model Checkpoint

The frozen production model checkpoint is stored in [NeuroADHD_ModelBenchmark/results/final_locked/](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/results/final_locked/):
- **Model Checkpoint**: `final_model.joblib` (74,409 bytes)
- **Architecture**: `sklearn.ensemble.RandomForestClassifier`
- **Key Hyperparameters**:
  - `n_estimators`: `50`
  - `max_depth`: `3`
  - `class_weight`: `'balanced'`
  - `max_features`: `'sqrt'` (10 features per split candidate)
  - `random_state`: `42`
- **Preprocessing Pipeline**: `final_preprocessing_pipeline.joblib` (`SimpleImputer(median)` $\to$ `VarianceThreshold(0.0)` $\to$ `StandardScaler()` $\to$ `SelectKBest(f_classif, k=100)`).
- **Test Set Double-Gating**: `run_all_models.py` requires both `--evaluate-test` AND `--confirm-test-eval` flags to prevent accidental test-set contamination.

---

## 📈 Benchmark & Validation Results

Every metric reported below is derived directly from reproducibility artifacts and evaluated under strict leak-free protocols.

### 1. Peking_2 Held-Out Test Evaluation ($N=20$)
*Source Files: [NeuroADHD_ModelBenchmark/results/final_locked/final_test_results.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/results/final_locked/final_test_results.json), [final_confusion_matrix.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/results/final_locked/final_confusion_matrix.json), [benchmarkData.js](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelDashboard/frontend/src/data/benchmarkData.js)*

* **Sample Size & Split**: 20 subjects (10 ADHD, 10 Typically Developing Controls) from Peking University (`Peking_2`).
* **Evaluation Protocol Context**: This held-out test set represents a **same-institution, same-scanner holdout** (scanned on the same 3T Siemens Trio scanner as Peking_1 and Peking_3 training partitions). The test set was scored 10 times in total across project history (9 baseline development runs with identical results, 1 final model run); no hyperparameter tuning used it.
* **Accuracy**: **`60.00%`** (12/20 correct), Wilson score 95% CI: **`[38.7%, 78.1%]`**.
* **ROC-AUC**: **`0.5600`**, Stratified bootstrap 95% CI: **`[0.290, 0.830]`** (2,000 resamples).
* **PR-AUC**: **`0.6341`** (vs chance baseline of `0.5000`).
* **2x2 Confusion Matrix**:
  - True Positives (ADHD detected): **`6`**
  - False Negatives (ADHD missed): **`4`**
  - False Positives (Control misclassified): **`4`**
  - True Negatives (Control detected): **`6`**
  - Balanced Accuracy: **`60.00%`** | Sensitivity (Recall): **`60.00%`** | Specificity: **`60.00%`** | Precision: **`60.00%`** | F1-Score: **`0.6000`**.

---

### 2. Multi-Site Generalization (Leave-One-Institution-Out CV)
*Source File: [NeuroADHD_ModelDashboard/frontend/src/data/siteConfoundData.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelDashboard/frontend/src/data/siteConfoundData.json)*

To assess cross-scanner generalization, models were evaluated with Leave-One-Institution-Out (LOIO) cross-validation where all subjects from an entire institution are held out as the test set:

#### Full Cohort ($N=153$ subjects across 6 institutions: KKI, NYU, NeuroIMAGE, OHSU, Peking, Pittsburgh)
*Note: OHSU ($N=14$) and Pittsburgh ($N=21$) contributed 0 ADHD cases to the CC200 preprocessed release.*
- **Majority-Class Baseline**: Accuracy **`67.32%`** (Control: 103/153), Balanced Accuracy `50.00%`, ROC-AUC `0.5000`.
- **Site-Only Model** (logistic regression on one-hot site labels): In-sample AUC **`0.7221`**, Stratified 5-fold AUC **`0.6745`**, LOIO AUC **`0.3326`**.
- **Age/Sex-Only Baseline** (standardized age + sex covariates): LOIO AUC **`0.4457`** [0.359, 0.547], LOIO Accuracy **`0.5033`** [0.418, 0.582].
- **Functional Connectivity Alone** (Pearson $r \to$ Fisher-$z$, SelectKBest $k=100$, Balanced RF): LOIO AUC **`0.5342`** [0.439, 0.633], LOIO Accuracy **`0.5882`** [0.510, 0.667], Within-institution permutation **`p = 0.0859`** (not statistically significant).
- **Connectivity + Demographic Covariates**: LOIO AUC **`0.5025`** [0.412, 0.600], LOIO Accuracy **`0.5752`** [0.497, 0.647].

#### Clean Cohort ($N=118$ subjects across 4 institutions: KKI, NYU, NeuroIMAGE, Peking)
*Note: Excludes zero-ADHD sites (OHSU, Pittsburgh); merges Peking 1/2/3 into a single Peking institution.*
- **Majority-Class Baseline**: Accuracy **`57.63%`** (Control: 68/118), Balanced Accuracy `50.00%`, ROC-AUC `0.5000`.
- **Site-Only Model** (logistic regression on one-hot site labels): In-sample AUC **`0.5791`**, Stratified 5-fold AUC **`0.5010`**, LOIO AUC **`0.4609`**.
- **Age/Sex-Only Baseline** (standardized age + sex covariates): LOIO AUC **`0.5976`** [0.492, 0.702], LOIO Accuracy **`0.5763`** [0.492, 0.661].
- **Functional Connectivity Alone** (Pearson $r \to$ Fisher-$z$, SelectKBest $k=100$, Balanced RF): LOIO AUC **`0.6221`** [0.515, 0.724], LOIO Accuracy **`0.6186`** [0.534, 0.703], Within-institution permutation **`p = 0.0210`**.
- **Connectivity + Demographic Covariates**: LOIO AUC **`0.6218`** [0.516, 0.724], LOIO Accuracy **`0.6271`** [0.542, 0.712].

---

### 3. Apples-to-Apples Model Comparison (118-Subject Clean Cohort, LOIO)
*Source Files: [NeuroADHD_ModelBenchmark/experiments/model_comparison/model_comparison_results.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/experiments/model_comparison/model_comparison_results.json), [modelComparisonData.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelDashboard/frontend/src/data/modelComparisonData.json)*

All pipelines below were evaluated identically across the 4 LOIO folds on the 118-subject clean cohort:

| Pipeline Architecture | LOIO ROC-AUC [95% CI] | LOIO Accuracy [95% CI] | LOIO Balanced Acc [95% CI] | Permutation $p$-value |
| :--- | :--- | :--- | :--- | :--- |
| **Majority Class Baseline** | `0.5000` [N/A] | `57.63%` [N/A] | `50.00%` [N/A] | $p = 1.0000$ |
| **Demographics Only** (`followup_study`: nested selection on age+sex) | `0.5976` [0.492, 0.702] | `57.63%` [49.2%, 66.1%] | `57.68%` [48.9%, 66.4%] | N/A |
| **Demographics Only** (`final_candidate`: fixed balanced LogReg) | `0.5857` [0.489, 0.685] | `61.02%` [52.5%, 69.5%] | `63.26%` [55.1%, 71.0%] | $p = 0.0498$ |
| **Locked Production Pipeline** (SelectKBest $k=100$ + Balanced RF) | `0.6250` [0.518, 0.726] | `61.86%` [53.4%, 70.3%] | `60.82%` [51.8%, 69.9%] | $p = 0.0100$ |
| **Final Candidate Ensemble** (Nested 126-configuration CV grid) | `0.6253` [0.521, 0.728] | `58.47%` [48.3%, 67.8%] | `56.03%` [46.8%, 64.7%] | $p = 0.0050$* |

*\*Note: $p = 0.0050$ is the minimum possible permutation p-value achievable under $B = 200$ permutations.*

#### Paired Statistical Significance Against Demographics:
- **Locked RF vs. Demographics (Age/Sex)**: Mean $\Delta \text{AUC} = +0.0397$, 95% bootstrap CI: `[-0.1061, +0.1695]`, paired permutation **`p = 0.2670`** (not statistically significant).
- **Candidate Ensemble vs. Demographics**: Mean $\Delta \text{AUC} = +0.0400$, 95% bootstrap CI: `[-0.1038, +0.1714]`, paired permutation **`p = 0.2685`** (not statistically significant).
- **Candidate Ensemble vs. Locked RF**: Mean $\Delta \text{AUC} = +0.0003$, 95% bootstrap CI: `[-0.0633, +0.0639]`, paired permutation **`p = 0.4960`** (no detectable difference between architectures).

---

### 4. What Was Tried That Did Not Help
Extensive ablation studies and architectural alternatives were systematically benchmarked but failed to yield improvements:

1. **Class Balancing Techniques**: SMOTE, RandomUnderSampler, and artificial oversampling strategies did not reliably improve out-of-site generalization compared to native balanced class weights ([experiments/balancing_study/REPORT.md](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/experiments/balancing_study/REPORT.md)).
2. **Alternative Model Families**: Linear SVM, RBF-kernel SVM, ExtraTrees, and gradient boosted trees (XGBoost) did not outperform a simple shallow Random Forest under multi-site cross-validation ([runs/full_benchmark_01/run_report.json](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/runs/full_benchmark_01/run_report.json)).
3. **Partial Correlation Matrices**: Precision matrix estimation using Ledoit-Wolf shrinkage failed to outperform standard Fisher $z$-transformed Pearson correlation matrices under LOIO evaluation ([experiments/final_candidate/REPORT.md](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/experiments/final_candidate/REPORT.md)).
4. **Riemannian Tangent Space Embeddings**: Projecting symmetric positive-definite covariance matrices onto the Riemannian manifold tangent space yielded negligible performance changes relative to flat Pearson connectivity ([experiments/final_candidate/REPORT.md](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/experiments/final_candidate/REPORT.md)).
5. **Nested Hyperparameter Grid Search**: Exhaustive inner-loop optimization over 126 pipeline configurations (varying correlation metric, feature selection cutoff $k \in \{50, 100, 200, 300\}$, and classifier depth) converged to parsimonious depth-3 trees and did not statistically improve generalization over the locked baseline ($\Delta \text{AUC} = +0.0003, p = 0.4960$) ([experiments/final_candidate/REPORT.md](file:///Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/experiments/final_candidate/REPORT.md)).

---

### 5. What This Shows / Does Not Show

> [!IMPORTANT]
> #### Interpretation & Methodological Takeaways
>
> **What this benchmark shows:**
> - Resting-state functional connectivity did **not** demonstrate a statistically significant performance advantage over simple age and sex demographic covariates on this cohort (Locked RF $\Delta \text{AUC} = +0.0397, p = 0.2670$; Ensemble $\Delta \text{AUC} = +0.0400, p = 0.2685$).
> - Age and biological sex carry substantial label-predictive variance in this dataset (LOIO AUC $0.5857 - 0.5976$), reflecting demographic composition differences between clinical and control recruitment pools rather than an ADHD-specific neural signature.
> - The 118-subject clean cohort was selected post-hoc (exploratory) after the full 153-subject cohort showed no statistically significant connectivity signal ($p = 0.0859$) due to severe site imbalances (e.g., OHSU and Pittsburgh containing zero ADHD cases).
> - Both pipelines (locked RF and nested candidate ensemble) were tuned with prior awareness of these subject pools.
>
> **What this benchmark does not show:**
> - It does **not** demonstrate that ADHD cannot be classified from functional neuroimaging.
> - It indicates only that this specific sample ($N=118\text{--}153$), preprocessing pipeline (C-PAC CC200 atlas), and feature representation do not establish generalizable diagnostic signal beyond basic demographics.
> - This software is strictly an academic research exploration and is **not** a clinical diagnostic tool.

---

### 6. Recommended Next Steps
- **Scanner Harmonization (ComBat / neuroCombat)**: Empirically evaluate empirical Bayes harmonization on multi-site cohorts with balanced site distributions (untested in this benchmark due to out-of-site CV constraints where held-out sites are unseen).
- **Large Prospective Cohorts**: Standardized multi-site acquisition protocols with larger samples (e.g., ABCD Study, $N > 10,000$).
- **Multimodal Integration**: Combining resting-state connectivity with structural T1w volumetrics, DTI white-matter tractography, and cognitive assessments.

---

## 🧪 Testing & Verification

Run the full validation suite to verify zero-leakage constraints, API functionality, and test-set locking:

```bash
# 1. Run core preprocessing & inference tests (5 tests)
./NeuroADHD/.venv/bin/python -m unittest discover -s NeuroADHD/tests

# 2. Run test-set locking protection tests (5 tests)
./NeuroADHD/.venv/bin/python -m unittest NeuroADHD_ModelBenchmark/tests/test_test_set_locking.py

# 3. Run dashboard backend API tests (8 tests)
./NeuroADHD/.venv/bin/python -m unittest discover -s NeuroADHD_ModelDashboard/backend/tests

# 4. Verify frontend production build
cd NeuroADHD_ModelDashboard/frontend && npm run build
```

---

## 📚 Data and Citation

- **ADHD-200 Sample**: ADHD-200 Consortium, shared via INDI/NITRC (https://fcon_1000.projects.nitrc.org/indi/adhd200/).
- **Preprocessed Data**: Neuro Bureau (CC200 via C-PAC).
- **Access and Usage**: Non-commercial research use. A free NITRC account is needed to download.
- **Redistribution Policy**: No raw scans, ROI time series, or connectivity feature matrices are included; per-subject model outputs (predictions and attributions) keyed by public ADHD-200 IDs are included for reproducibility.
- **Citation**: Please cite the ADHD-200 Consortium and the Neuro Bureau.
