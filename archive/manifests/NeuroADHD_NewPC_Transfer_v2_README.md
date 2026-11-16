# NeuroADHD New-PC Transfer Package

## What this package contains

This package preserves the current NeuroADHD research workspace for transfer. It includes the CC200 data/features and code, every discovered serialized benchmark estimator and fitted preprocessing pipeline, historical experiments, dashboard source and build, official ADHD-200 audit/expansion records, the verified 18-subject raw pilot, and an additional already-present partial RawDataBIDS subset.

No model was retrained, no MRI was preprocessed, and the locked test was not re-evaluated. Model artifacts are byte-for-byte copies. Mac virtual environments, benchmark model_envs, node_modules, caches, .DS_Store files, bytecode, and temporary PDF rendering output were omitted.

## Directory guide

- NeuroADHD/ — project code, requirements, phenotype/site/diagnosis mapping, CC200 time courses, motion/phenotype inputs, feature metadata/QC, 153-subject feature matrix, and split manifests.
- NeuroADHD_ModelBenchmark/ — training/evaluation scripts, configs, every run directory, model files, logs, metrics, predictions, reports, explainability, and CC200 mapping.
- NeuroADHD_ModelDashboard/ — React/Vite source and lockfile, built static frontend at frontend/dist/, backend source, screenshots, and illustrated PDF guide.
- ADHD200_Official/ — official provenance/source inventories, phenotype/derivative audits, download records, historical C-PAC recovery report, modern pipeline proposal, and raw pilot.
- ADHD200_RawDataBIDS_subset/ — additional partial Brown/KKI RawDataBIDS files outside the project folders, clearly separated from the 18-subject pilot.
- NeuroADHD_ModelInventory.csv — every discovered serialized model and pipeline, plus report-level candidates whose fitted binary is absent.
- PACKAGE_MANIFEST.csv and PACKAGE_MANIFEST.json — archived file paths, original paths, sizes, SHA-256 hashes, categories, inclusion reasons, and workflow flags.
- checksums.sha256 — SHA-256 values for archived payload and manifest files. It does not hash itself.

## Current 153-subject CC200 workflow

Feature matrix:
NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv

It has 153 rows and 19,903 columns: subject_id, site, 19,900 ROI-pair features, and diagnosis. Control = 0 (103); ADHD = 1 (50). The source feature ordering is preserved and described by cc200_feature_metadata.csv.

The subject-label mapping and QC reports are under NeuroADHD/data/adhd200/interim/. The authoritative split is NeuroADHD/data/adhd200/processed/splits/subject_split_manifest.csv: 114 training subjects, 19 NeuroIMAGE validation subjects, 20 Peking_2 held-out subjects. Companion subject lists and validation reports are in that directory.

## Saved model files and model inventory

The selected historical Balanced Random Forest is present at:

- NeuroADHD_ModelBenchmark/results/final_locked/final_model.joblib
- NeuroADHD_ModelBenchmark/results/final_locked/final_preprocessing_pipeline.joblib

A matching run-specific copy is under NeuroADHD_ModelBenchmark/runs/run_20261004_182318_final_locked/. The final run includes the expected results JSON, predictions CSV, confusion matrix JSON, model metadata JSON, and Markdown/HTML report. The benchmark reports directory also includes the final locked-test report.

The scan found 136 serialized files: 68 estimator files and 68 fitted preprocessing pipelines. All run-specific copies were retained, including byte-identical legacy models/ and newer runs/ copies. They have 9 distinct SHA-256 payloads. File presence, size, source/stage hash, and package checksums passed. Joblib objects were not deserialized; Windows load compatibility and exact dependency compatibility remain to be tested.

The inventory also records Opt-1/Opt-2 selected candidates, hard/soft/stacking results, repeated hybrid groups, and RF seeds 47-49 for which reports exist but no corresponding fitted binary was found. Missing rows summarize report-level candidates, not every transient CV/grid-search fold estimator.

## A. Load an existing model for inference

The saved estimator and fitted preprocessing pipeline are separate. For a new subject, first calculate the same 19,900 CC200 Fisher-z values in the exact feature order. Example usage:

~~~python
from pathlib import Path
import joblib
import pandas as pd

root = Path(".")
reference_csv = root / "NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv"
model_path = root / "NeuroADHD_ModelBenchmark/results/final_locked/final_model.joblib"
preprocessor_path = root / "NeuroADHD_ModelBenchmark/results/final_locked/final_preprocessing_pipeline.joblib"

schema = pd.read_csv(reference_csv, nrows=0)
feature_cols = [c for c in schema.columns if c.startswith("ROI_")]
new_rows = pd.read_csv("new_subject_connectivity.csv")
X_new = new_rows.loc[:, feature_cols].to_numpy(dtype="float64")

preprocessor = joblib.load(preprocessor_path)
model = joblib.load(model_path)
X_ready = preprocessor.transform(X_new)
adhd_probability = model.predict_proba(X_ready)[:, 1]
predicted_label = model.predict(X_ready)
~~~

The example input new_subject_connectivity.csv must be generated separately. It is not included or fabricated. Predictions are research outputs, not diagnoses. Check library versions before loading pickle-based artifacts.

## B. Retrain from existing connectivity features

Keep NeuroADHD/ and NeuroADHD_ModelBenchmark/ as sibling directories after extraction. The single-model script accepts --model and --run-id. From NeuroADHD_ModelBenchmark/, on Windows:

~~~powershell
..\NeuroADHD\.venv\Scripts\python.exe scripts\train_model.py --model random_forest --run-id newpc_rf_retrain
~~~

On macOS/Linux:

~~~bash
../NeuroADHD/.venv/bin/python scripts/train_model.py --model random_forest --run-id newpc_rf_retrain
~~~

This fits preprocessing on the training partition and tunes with validation. Use a fresh run ID. The runner also has a verified --no-test option to run configured models while leaving the locked test unevaluated:

~~~powershell
..\NeuroADHD\.venv\Scripts\python.exe scripts\run_all_models.py --run-id newpc_all_models --no-test
~~~

Do not run a final-test script unless you intend a new evaluation of the locked Peking_2 subjects under a predeclared protocol.

## C. Recreate CC200 features

The extractor source is NeuroADHD/src/features/extract_cc200_connectivity.py. It reads raw roi_CC200.csv files plus the validated label mapping and writes to the existing feature output path. Back up the feature table or work on a copy before running it. Packaging did not run the extractor or change the source data.

## D. Process raw MRI for new subjects

The official pilot is ADHD200_Official/experiments/raw_pilot_18/raw/BIDS/. The pilot manifest lists 18 subjects, two at each of nine sites. All 85 successful download-manifest entries are present and match expected sizes. Raw technical validation records 18/18 passed. Preprocessing is 0/18, CC200 extraction is 0/18, and connectivity extraction is 0/18.

ADHD200_RawDataBIDS_subset/BIDS/ contains 137 complete Brown/KKI source files matching the official object inventory by path and size. Three interrupted .part files are listed in EXCLUDED_INCOMPLETE_DOWNLOADS.csv and were excluded. The source listing describes 2,424 objects / 81.89 GiB overall; only local files are packaged. This is not the full raw collection.

Historical C-PAC is a known blocker: the audit identifies CPAC 0.3.9.1 and some settings, but the run-specific YAML/data config, exact launch command, complete dependencies, and verifiable original container digest are missing. Exact historical reproduction cannot be claimed. The modern C-PAC 1.8.6 profile is a proposal, not an executed/validated pipeline. Packaging does not resolve this blocker.

## Python environment on the new PC

No virtual environment was copied. NeuroADHD/requirements.txt is the available pinned dependency specification from the source Python 3.11 environment. It may include platform-specific packages and does not guarantee Windows wheel availability. Per-run environment version records are included where present. The benchmark model_envs were copied neither as environments nor lockfiles.

Create a new Python 3.11 environment on Windows:

~~~powershell
py -3.11 -m venv NeuroADHD\.venv
NeuroADHD\.venv\Scripts\python.exe -m pip install --upgrade pip
NeuroADHD\.venv\Scripts\python.exe -m pip install -r NeuroADHD\requirements.txt
~~~

If a pin has no Windows wheel, compare the run-specific environment record and document any deliberate version change; do not silently change versions for old serialized models.

Run the included tests:

~~~powershell
NeuroADHD\.venv\Scripts\python.exe -m pytest NeuroADHD\tests
~~~

Restore the dashboard frontend using its lockfile:

~~~powershell
cd NeuroADHD_ModelDashboard\frontend
npm ci
npm run dev
~~~

Open the URL printed by Vite. The existing built frontend is under frontend/dist/ and can be served with:

~~~powershell
python -m http.server 8080 --directory NeuroADHD_ModelDashboard\frontend\dist
~~~

Backend source is included, but no backend-specific dependency manifest was found; the dashboard README refers to FastAPI/Uvicorn through the Mac Python environment. Backend startup and API paths require destination-PC review. node_modules was not copied.

## Provenance and verification

PACKAGE_MANIFEST.csv/.json list substantive archive payload files with original path, size, SHA-256, category, reason, and workflow flags. The three bookkeeping files (the two manifests and checksums.sha256) cannot list their own exact checksum without a circular reference; checksums.sha256 covers both manifests, and the external checksum file covers the ZIP and root-level deliverables.

NeuroADHD_ModelInventory.csv records all found serialized files and report-level missing candidates. Source files were copied without editing; all copy-time hashes matched. No credentials or private keys were included. Incomplete .part downloads and .DS_Store files were excluded.

## Not done in this packaging task

- No training, feature extraction, MRI preprocessing, or test-set evaluation.
- No Windows load test for serialized joblib objects.
- No recovery of the historical C-PAC run configuration/environment.
- No download of the remaining ADHD-200 participants.
