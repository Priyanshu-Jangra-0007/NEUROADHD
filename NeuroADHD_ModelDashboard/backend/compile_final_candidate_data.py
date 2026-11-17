#!/usr/bin/env python3
"""
Compile final_candidate benchmark results into a dedicated frontend JSON artifact.
Generates NeuroADHD_ModelDashboard/frontend/src/data/finalCandidateData.json
directly from NeuroADHD_ModelBenchmark/experiments/final_candidate/candidate_study_results.json.
"""

import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent.parent
    src_file = root / "NeuroADHD_ModelBenchmark" / "experiments" / "final_candidate" / "candidate_study_results.json"
    dst_file = root / "NeuroADHD_ModelDashboard" / "frontend" / "src" / "data" / "finalCandidateData.json"
    
    if not src_file.exists():
        raise FileNotFoundError(f"Source candidate results not found at {src_file}")
        
    with open(src_file) as f:
        data = json.load(f)
        
    m = data["loio"]["metrics"]
    ci = data["loio"]["ci"]
    maj = data["metadata"]["majority_accuracy"]
    perm = data["permutation_significance"]["final_model"]
    cov_m = data["covariates_only_loio"]["metrics"]
    cov_ci = data["covariates_only_loio"]["ci"]
    cov_perm = data["permutation_significance"]["covariates_only_model"]
    paired = data["paired_bootstrap_vs_covars"]
    
    payload = {
        "_metadata": {
            "source": "NeuroADHD_ModelBenchmark/experiments/final_candidate/candidate_study_results.json",
            "study": "Comprehensive Final Candidate Benchmark (126-Configuration Space)",
            "cohort": "118_filtered (Excluding OHSU & Pittsburgh)",
            "n_subjects": data["metadata"]["n_subjects"]
        },
        "majority_accuracy": float(maj),
        "loio": {
            "roc_auc": float(m["roc_auc"]),
            "roc_auc_ci": [float(ci["roc_auc"][0]), float(ci["roc_auc"][1])],
            "accuracy": float(m["accuracy"]),
            "accuracy_ci": [float(ci["accuracy"][0]), float(ci["accuracy"][1])],
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "balanced_accuracy_ci": [float(ci["balanced_accuracy"][0]), float(ci["balanced_accuracy"][1])],
            "sensitivity": float(m["sensitivity"]),
            "sensitivity_ci": [float(ci["sensitivity"][0]), float(ci["sensitivity"][1])],
            "specificity": float(m["specificity"]),
            "specificity_ci": [float(ci["specificity"][0]), float(ci["specificity"][1])],
            "precision": float(m["precision"]),
            "mcc": float(m["mcc"]),
            "confusion_matrix": {
                "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"]
            }
        },
        "permutation_test": {
            "n_permutations": perm.get("n_permutations", 200),
            "observed_auc": float(perm["observed_auc"]),
            "null_mean": float(perm["null_mean"]),
            "null_std": float(perm["null_std"]),
            "p_value": float(perm["p_value"]),
            "p_value_display": "p = 0.0050",
            "is_minimum_possible": True,
            "formula_explanation": "Minimum possible p-value with 200 permutations: (0 + 1) / (200 + 1) = 0.004975 (~0.0050), indicating zero null iterations exceeded the observed AUC."
        },
        "covariates_only_model": {
            "description": "Standardized age and sex demographic covariates alone (Logistic Regression)",
            "loio_auc": float(cov_m["roc_auc"]),
            "loio_auc_ci": [float(cov_ci["roc_auc"][0]), float(cov_ci["roc_auc"][1])],
            "loio_accuracy": float(cov_m["accuracy"]),
            "loio_accuracy_ci": [float(cov_ci["accuracy"][0]), float(cov_ci["accuracy"][1])],
            "p_value": float(cov_perm["p_value"])
        },
        "paired_bootstrap_vs_covars": {
            "delta_mean": float(paired["delta_mean"]),
            "delta_ci_95": [float(paired["delta_ci_95"][0]), float(paired["delta_ci_95"][1])],
            "p_value": float(paired["p_value_one_sided"]),
            "is_significant": False,
            "conclusion": "Non-significant (p = 0.2685, 95% CI spans zero: [-0.1038, +0.1714]). Functional connectivity does not statistically outperform demographic covariates alone."
        },
        "selected_configurations": {
            "differ_by_fold": True,
            "folds": data["loio"]["selected_configs"]
        },
        "limitations": [
            {
                "title": "Meta-Selection Confound",
                "detail": "The 118-subject clean subset and hyperparameter search space (Pearson/Partial/Tangent, k in {50, 100, 200}) were curated after exploratory analyses on the same data."
            },
            {
                "title": "Permutation Test Conditions on Narrowed Grid",
                "detail": "The permutation test validates that the selected pipeline beats chance for the narrowed 126-configuration grid, but does not repeat the initial space exploration or cohort selection."
            },
            {
                "title": "Non-Significant Separation from Demographics",
                "detail": "While the final candidate achieves LOIO AUC 0.6253 (p=0.0050), the paired bootstrap difference against the Age/Sex-only baseline (ΔAUC = +0.0400 [-0.1038, +0.1714], p=0.2685) is not statistically significant."
            }
        ]
    }
    
    with open(dst_file, "w") as f:
        json.dump(payload, f, indent=2)
        
    print(f"Successfully compiled {dst_file} ({dst_file.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
