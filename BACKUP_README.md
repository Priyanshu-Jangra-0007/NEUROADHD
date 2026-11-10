# NeuroADHD: Portable Backup & Migration Package (FINAL)

**Archive Name:** `NeuroADHD_Portable_Backup_FINAL.zip`  
**Generated Date:** 2026-10-21  
**Architecture:** Python 3.11+ (FastAPI, Scikit-Learn, XGBoost) & React 18 / Vite  

---

## 1. Package Overview

This archive contains the complete, self-contained, and permanent source code, trained model checkpoints, experiment history, dataset manifests, CC200 functional connectivity feature matrices, and interactive dashboard for the **NeuroADHD** machine learning platform.

### What Is Included in This Archive:
* **`NeuroADHD/`**: Core feature engineering pipeline, BIDS/C-PAC dataset manifests, CC200 functional connectivity extractor (`19,900` features per subject), site-disjoint split manifests, unit tests for leakage prevention, and raw CC200 time-series data for all 153 subjects.
* **`NeuroADHD_ModelBenchmark/`**: Full model benchmark suite containing:
  * 7 model configurations (`logistic_regression`, `linear_svm`, `rbf_svm`, `random_forest`, `balanced_random_forest`, `extra_trees`, `xgboost`).
  * Advanced runners: Voting & Stacking Ensembles, K-feature selection ( \in \{50, 100, 200, 500\}$), Repeated Stratified CV, and SHAP explainability.
  * All historical runs under `runs/` including the latest `balanced_rf_01` and `full_benchmark_01`.
  * All 164 trained `.joblib` model checkpoints under `models/` and `results/`.
* **`NeuroADHD_ModelDashboard/`**: Full-stack decision support dashboard:
  * FastAPI backend (`server.py`) with service layer (`benchmark_service.py`).
  * React 18 + Vite frontend with 20 analytical views, Recharts ROC/PR curves, brain network graphs, and subject diagnostics.
  * `package.json` and `package-lock.json` for reproducible dependency installation.
* **`ADHD200_Official/`**: Official ADHD-200 Consortium phenotypic manifests, metadata dictionaries, S3 bucket audit logs, and pilot experiment records.
* **`outputs/`**: Local raw BIDS NIfTI MRI data subset and manifests (`outputs/ADHD200_RawDataBIDS/`).
* **Root Inventories & Reports**: `BACKUP_MANIFEST.csv`, `NeuroADHD_NewPC_ModelInventory_v2.csv`, `NeuroADHD_Workspace_Architecture_and_Dataset_Analysis.pdf`, `generate_pdf_report.py`, and `work_download_adhd200.py`.

### What Is Excluded (Reproducible Environment Artifacts):
* Python virtual environments (`.venv/`, `model_envs/`) - must be recreated natively on the target machine.
* Node dependencies (`node_modules/`) - installed via `npm install`.
* Git internals (`.git/`) and temporary caches (`tmp/`, `dist/`, `__pycache__/`).
* Duplicate old transfer ZIPs.

---

## 2. Quickstart & Setup on a New Mac (Apple Silicon / Intel)

### Step 1: Extract the Archive
```bash
mkdir -p NeuroADHD_Project
cd NeuroADHD_Project
unzip /path/to/NeuroADHD_Portable_Backup_FINAL.zip
```

### Step 2: System Dependencies (macOS / Apple Silicon)
Ensure Xcode command line tools and Homebrew are installed:
```bash
xcode-select --install
brew install libomp  # Required for XGBoost on macOS Apple Silicon (arm64)
```

### Step 3: Set Up Python Virtual Environment
```bash
# Create and activate virtual environment inside NeuroADHD
cd NeuroADHD
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Return to project root
cd ..
```

### Step 4: Set Up Frontend Dependencies
```bash
cd NeuroADHD_ModelDashboard/frontend
npm install
cd ../..
```

---

## 3. Launching Services

### 1. Launch FastAPI Backend (Port 8000)
```bash
cd NeuroADHD_ModelDashboard/backend
../../NeuroADHD/.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
* API Documentation: `http://localhost:8000/docs`
* System Status: `http://localhost:8000/api/status`

### 2. Launch React Frontend (Port 5173)
```bash
cd NeuroADHD_ModelDashboard/frontend
npm run dev
```
* Dashboard URL: `http://localhost:5173`

---

## 4. Running Model Benchmark & Training

### Train a Single Model:
```bash
cd NeuroADHD_ModelBenchmark
mkdir -p runs/rf_run_01
../NeuroADHD/.venv/bin/python scripts/train_model.py --model balanced_random_forest --run-id rf_run_01
```

### Run Repeated Cross-Validation Stability Test (Recommended):
```bash
cd NeuroADHD_ModelBenchmark
mkdir -p runs/rf_repeated_cv_02
../NeuroADHD/.venv/bin/python scripts/run_rf_repeated_cv.py --run-id rf_repeated_cv_02
```

### Run Feature Selection Optimization (K=50, 100, 200, 500):
```bash
cd NeuroADHD_ModelBenchmark
mkdir -p runs/focused_opt_02
../NeuroADHD/.venv/bin/python scripts/run_focused_optimization.py --run-id focused_opt_02
```

---

## 5. Dataset Architecture & Split Protocol

* **Total Cohort:** 153 subjects from 8 ADHD-200 Consortium sites (103 Control, 50 ADHD).
* **Feature Space:** Craddock 200 (CC200) atlas $ightarrow$  	imes 199 / 2 = 19,900$ Fisher-z functional connectivity edges per subject.
* **Site-Disjoint Validation Design:**
  * **Train Partition (=114$):** KKI (24), NYU (6), OHSU (14), Peking_1 (38), Peking_3 (11), Pittsburgh (21).
  * **Validation Partition (=19$):** NeuroIMAGE (19).
  * **Locked Test Partition (=20$):** Peking_2 (20) — held out to evaluate real-world clinical generalizability.

---

## 6. Model Benchmark Leaderboard

| Model Architecture | Val ROC-AUC | Val Accuracy | Notes |
|---|---|---|---|
| **Random Forest (Balanced, K=100)** | **0.7000** | **68.42%** | **Best balanced individual model** (`run_20261004_182318_final_locked`) |
| **Soft Voting (0.6 RF + 0.4 XGB)** | **0.6889** | **68.42%** | **Best ensemble model** (`run_20261004_182834_ensemble`) |
| **Balanced RF (8-Rep CV)** | **0.6045** | **64.47%** | High stability across random seeds (`run_20261004_183929_3more_rf_repeated`) |
| **Random Forest (Balanced Subsample)**| **0.7000** | **52.63%** | +10.5% accuracy gain over raw baseline (`balanced_rf_01`) |
| **Baseline Random Forest (Unweighted)**| **0.7333** | **42.11%** | Baseline on raw 19,900 features (`full_benchmark_01`) |
| **Locked Test Evaluation (Peking_2)** | **0.5600** | **60.00%** | Unseen scanner cohort validation |

---

## 7. Checksums & Verification

All files are cataloged with exact byte counts and SHA-256 hashes in `BACKUP_MANIFEST.csv`.  
To verify integrity after transfer:
```bash
sha256sum -c BACKUP_MANIFEST.csv  # on Linux
shasum -a 256 -c BACKUP_MANIFEST.csv  # on macOS
```
