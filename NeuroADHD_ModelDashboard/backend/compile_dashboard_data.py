#!/usr/bin/env python3
"""
Compile authoritative NeuroADHD benchmark, robustness, explainability, and atlas mapping data
into frontend data modules for the Research Dashboard.
"""

import json
import pandas as pd
from pathlib import Path

BENCHMARK_ROOT = Path("/Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark")
DATA_DIR = Path("/Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD/data/adhd200")
FRONTEND_DATA_DIR = Path("/Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelDashboard/frontend/src/data")
FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Final Locked Test Results
final_run_dir = BENCHMARK_ROOT / "runs/run_20261004_182318_final_locked"
with open(final_run_dir / "final_test_results.json") as f:
    final_test_results = json.load(f)
with open(final_run_dir / "final_confusion_matrix.json") as f:
    final_confusion_matrix = json.load(f)
with open(final_run_dir / "final_model_metadata.json") as f:
    final_model_metadata = json.load(f)

# 2. Repeated RF Robustness
rf_rep_dir = BENCHMARK_ROOT / "runs/run_20261004_183929_3more_rf_repeated"
with open(rf_rep_dir / "rf_summary_statistics.json") as f:
    rf_summary_stats = json.load(f)
df_rf_reps = pd.read_csv(rf_rep_dir / "rf_repeated_results.csv")
rf_reps_list = df_rf_reps.to_dict(orient="records")

# 3. Repeated Hybrid Comparison
hybrid_rep_dir = BENCHMARK_ROOT / "runs/run_20261004_183509_repeated_hybrid_01"
with open(hybrid_rep_dir / "summary_statistics.json") as f:
    hybrid_summary_stats = json.load(f)
with open(hybrid_rep_dir / "model_ranking.json") as f:
    hybrid_rankings = json.load(f)

# 4. Explainability Artifacts
exp_dir = BENCHMARK_ROOT / "runs/run_20261004_184658_explainability/explainability"
df_global_shap = pd.read_csv(exp_dir / "shap_global_importance.csv")
df_top_edges = pd.read_csv(exp_dir / "top_connectivity_edges.csv")
df_roi_hubs = pd.read_csv(exp_dir / "roi_importance_summary.csv")
df_sub_exp = pd.read_csv(exp_dir / "subject_explanations.csv")

# 5. Atlas Mapping Artifacts
mapping_dir = BENCHMARK_ROOT / "runs/run_20261004_185219_atlas_mapping/atlas_mapping"
df_mapped_edges = pd.read_csv(mapping_dir / "mapped_connectivity_edges.csv")
df_mapped_hubs = pd.read_csv(mapping_dir / "mapped_hub_rois.csv")
df_network_summary = pd.read_csv(mapping_dir / "network_connectivity_summary.csv")

# 6. Manifest & Cohort Breakdown
manifest = pd.read_csv(DATA_DIR / "processed/splits/subject_split_manifest.csv")
cohort_breakdown = {
    "total_subjects": len(manifest),
    "train_subjects": int((manifest["partition"] == "train").sum()),
    "train_adhd": int(((manifest["partition"] == "train") & (manifest["diagnosis"] == 1)).sum()),
    "train_control": int(((manifest["partition"] == "train") & (manifest["diagnosis"] == 0)).sum()),
    "validation_subjects": int((manifest["partition"] == "validation").sum()),
    "validation_adhd": int(((manifest["partition"] == "validation") & (manifest["diagnosis"] == 1)).sum()),
    "validation_control": int(((manifest["partition"] == "validation") & (manifest["diagnosis"] == 0)).sum()),
    "validation_site": "NeuroIMAGE",
    "test_subjects": int((manifest["partition"] == "test").sum()),
    "test_adhd": int(((manifest["partition"] == "test") & (manifest["diagnosis"] == 1)).sum()),
    "test_control": int(((manifest["partition"] == "test") & (manifest["diagnosis"] == 0)).sum()),
    "test_site": "Peking_2 (Locked)",
    "sites_count": int(manifest["site"].nunique()),
    "sites_list": sorted(manifest["site"].unique().tolist())
}

# Compile into master dashboard data object
dashboard_data = {
    "cohort_breakdown": cohort_breakdown,
    "final_model_metadata": final_model_metadata,
    "final_test_results": final_test_results,
    "final_confusion_matrix": final_confusion_matrix,
    "rf_robustness": {
        "summary": rf_summary_stats,
        "repetitions": rf_reps_list
    },
    "hybrid_comparison": {
        "summary": hybrid_summary_stats,
        "rankings": hybrid_rankings
    },
    "explainability": {
        "top_edges": df_mapped_edges.to_dict(orient="records"),
        "hub_rois": df_mapped_hubs.to_dict(orient="records"),
        "roi_summary": df_roi_hubs.head(30).to_dict(orient="records"),
        "subject_explanations": df_sub_exp.to_dict(orient="records"),
        "network_summary": df_network_summary.to_dict(orient="records")
    }
}

# Write JS module for direct import in React
js_code = f"export const BENCHMARK_DATA = {json.dumps(dashboard_data, indent=2)};\n"
with open(FRONTEND_DATA_DIR / "benchmarkData.js", "w") as f:
    f.write(js_code)

# Also write JSON format
with open(FRONTEND_DATA_DIR / "benchmarkData.json", "w") as f:
    json.dump(dashboard_data, f, indent=2)

print("Successfully compiled benchmarkData.js and benchmarkData.json for Research Dashboard!")
