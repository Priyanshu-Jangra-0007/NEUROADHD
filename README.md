# NeuroADHD: Resting-State fMRI CC200 Benchmark & Research Dashboard

An academic machine learning benchmark and exploration dashboard investigating functional connectivity patterns in Attention-Deficit/Hyperactivity Disorder (ADHD) using the ADHD-200 CC200 resting-state fMRI dataset.

> [!WARNING]
> **Research Use Only — Not a Clinical Diagnosis:**
> This software is an exploratory research benchmark for neuroimaging and machine learning methodologies. It is **not** an FDA-cleared or CE-marked medical diagnostic device. Model outputs reflect statistical pattern rankings and are **not calibrated clinical diagnostic probabilities**.

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

## ⚠️ Critical Scientific & Methodological Caveats

When presenting or interpreting findings from this benchmark, the following caveats must be disclosed:

1. **Same-Scanner Batch Holdout (Not True Out-of-Site Generalization)**:
   The 20 held-out test subjects in `results/final_locked/final_test_predictions.csv` originate exclusively from the Peking_2 partition. Because Peking_1 and Peking_3 subjects were used in model development and all three partitions were acquired on the exact same 3T Siemens Trio scanner at Peking University with identical acquisition protocols, the test set measures **same-scanner batch reproducibility**, not out-of-site generalization to novel institutions.

2. **Severe Multi-Site Confounding**:
   Across the 153 subjects in the CC200 dataset, diagnosis is strongly confounded with scanning site. Two institutions contributed zero ADHD cases:
   - **OHSU**: 23 Control, 0 ADHD
   - **Pittsburgh**: 12 Control, 0 ADHD
   
   Leave-One-Institution-Out (LOIO) cross-validation demonstrates that standard random $k$-fold cross-validation inflates performance metrics because models inadvertently learn site-specific scanner artifacts:
   - All 153 subjects LOIO AUC: **`0.5670 ± 0.158`**
   - 118 subjects LOIO AUC (excluding OHSU and Pittsburgh): **`0.5846 ± 0.125`**

3. **High Statistical Variance ($N=20$)**:
   The held-out test partition consists of only 20 individuals (10 ADHD, 10 Control). The test accuracy of $60.0\%$ (12/20 correct) corresponds to a Wilson 95% confidence interval of **`[38.7%, 78.1%]`**, which spans chance-level performance ($50\%$). The test ROC-AUC of $0.580$ has a stratified bootstrap 95% interval of **`[0.290, 0.830]`**.

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
