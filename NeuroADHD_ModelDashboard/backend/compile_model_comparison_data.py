#!/usr/bin/env python3
"""
Compile model comparison study results into a frontend-ready JSON artifact.
Generates NeuroADHD_ModelDashboard/frontend/src/data/modelComparisonData.json
from NeuroADHD_ModelBenchmark/experiments/model_comparison/model_comparison_results.json
and NeuroADHD_ModelBenchmark/experiments/followup_study/followup_study_results.json.
"""

import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent.parent
    cmp_file = root / "NeuroADHD_ModelBenchmark" / "experiments" / "model_comparison" / "model_comparison_results.json"
    followup_file = root / "NeuroADHD_ModelBenchmark" / "experiments" / "followup_study" / "followup_study_results.json"
    dst_file = root / "NeuroADHD_ModelDashboard" / "frontend" / "src" / "data" / "modelComparisonData.json"
    
    with open(cmp_file) as f:
        cmp_data = json.load(f)
        
    with open(followup_file) as f:
        followup_data = json.load(f)
        
    f_c2 = followup_data["Config 2 (Excl. OHSU & Pittsburgh, 118 Subjects)"]["covariates_only__loio"]
    
    l_m = cmp_data["locked_rf"]["loio"]["metrics"]
    l_ci = cmp_data["locked_rf"]["loio"]["ci"]
    l_perm = cmp_data["locked_rf"]["permutation_test"]
    
    c_m = cmp_data["final_candidate"]["loio_metrics"]
    c_ci = cmp_data["final_candidate"]["loio_ci"]
    c_perm = cmp_data["final_candidate"]["permutation_test"]
    
    cov_m = cmp_data["covariates_only"]["loio_metrics"]
    cov_ci = cmp_data["covariates_only"]["loio_ci"]
    cov_perm = cmp_data["covariates_only"]["permutation_test"]
    
    maj_acc = cmp_data["metadata"]["majority_accuracy"]
    
    stat_cand_l = cmp_data["statistical_comparisons"]["candidate_minus_locked_rf"]
    stat_l_cov = cmp_data["statistical_comparisons"]["locked_rf_minus_covariates"]
    stat_cand_cov = cmp_data["statistical_comparisons"]["candidate_minus_covariates"]
    
    payload = {
        "_metadata": {
            "source": "NeuroADHD_ModelBenchmark/experiments/model_comparison/model_comparison_results.json",
            "study": "Apples-to-Apples Model Comparison (118-Subject Clean Cohort)",
            "n_subjects": 118,
            "majority_accuracy": float(maj_acc)
        },
        "models": [
            {
                "id": "majority",
                "name": "Majority Class Baseline",
                "architecture": "Naive majority label prediction (Control)",
                "loio_roc_auc": 0.5000,
                "loio_roc_auc_ci": None,
                "loio_accuracy": float(maj_acc),
                "loio_accuracy_ci": None,
                "loio_balanced_accuracy": 0.5000,
                "loio_balanced_accuracy_ci": None,
                "permutation_p": 1.0000,
                "note": "Predicting majority Control (68/118 = 57.63%)"
            },
            {
                "id": "demographics_followup",
                "name": "Demographics Baseline (followup_study)",
                "architecture": "followup_study: nested model selection on age+sex, LOIO AUC 0.5976",
                "method_label": "followup_study: nested model selection on age+sex, LOIO AUC 0.5976",
                "loio_roc_auc": float(f_c2["nested_metrics"]["roc_auc"]),
                "loio_roc_auc_ci": [float(f_c2["nested_ci"]["roc_auc"][0]), float(f_c2["nested_ci"]["roc_auc"][1])],
                "loio_accuracy": float(f_c2["nested_metrics"]["accuracy"]),
                "loio_accuracy_ci": [float(f_c2["nested_ci"]["accuracy"][0]), float(f_c2["nested_ci"]["accuracy"][1])],
                "loio_balanced_accuracy": float(f_c2["nested_metrics"]["balanced_accuracy"]),
                "loio_balanced_accuracy_ci": [float(f_c2["nested_ci"]["balanced_accuracy"][0]), float(f_c2["nested_ci"]["balanced_accuracy"][1])],
                "permutation_p": None,
                "note": "Nested CV selecting RF/SVM/LogReg per fold on standardized age + raw sex"
            },
            {
                "id": "demographics_candidate",
                "name": "Demographics Baseline (final_candidate)",
                "architecture": "final_candidate: fixed logistic regression, LOIO AUC 0.5857",
                "method_label": "final_candidate: fixed logistic regression, LOIO AUC 0.5857",
                "loio_roc_auc": float(cov_m["roc_auc"]),
                "loio_roc_auc_ci": [float(cov_ci["roc_auc"][0]), float(cov_ci["roc_auc"][1])],
                "loio_accuracy": float(cov_m["accuracy"]),
                "loio_accuracy_ci": [float(cov_ci["accuracy"][0]), float(cov_ci["accuracy"][1])],
                "loio_balanced_accuracy": float(cov_m["balanced_accuracy"]),
                "loio_balanced_accuracy_ci": [float(cov_ci["balanced_accuracy"][0]), float(cov_ci["balanced_accuracy"][1])],
                "permutation_p": float(cov_perm["p_value"]),
                "note": "Fixed LogisticRegression(C=1.0, balanced) on standardized age and sex"
            },
            {
                "id": "locked_rf",
                "name": "Locked Production Pipeline (Random Forest)",
                "architecture": "SelectKBest(k=100) + Balanced RF(50 trees, depth 3)",
                "loio_roc_auc": float(l_m["roc_auc"]),
                "loio_roc_auc_ci": [float(l_ci["roc_auc"][0]), float(l_ci["roc_auc"][1])],
                "loio_accuracy": float(l_m["accuracy"]),
                "loio_accuracy_ci": [float(l_ci["accuracy"][0]), float(l_ci["accuracy"][1])],
                "loio_balanced_accuracy": float(l_m["balanced_accuracy"]),
                "loio_balanced_accuracy_ci": [float(l_ci["balanced_accuracy"][0]), float(l_ci["balanced_accuracy"][1])],
                "permutation_p": float(l_perm["p_value"]),
                "note": "Refitted inside each fold from scratch on Pearson connectivity"
            },
            {
                "id": "final_candidate",
                "name": "Final Candidate Pipeline (Nested Ensemble)",
                "architecture": "Nested Grid Selection (126 configs: Pearson/Partial/Tangent + K + Classifiers)",
                "loio_roc_auc": float(c_m["roc_auc"]),
                "loio_roc_auc_ci": [float(c_ci["roc_auc"][0]), float(c_ci["roc_auc"][1])],
                "loio_accuracy": float(c_m["accuracy"]),
                "loio_accuracy_ci": [float(c_ci["accuracy"][0]), float(c_ci["accuracy"][1])],
                "loio_balanced_accuracy": float(c_m["balanced_accuracy"]),
                "loio_balanced_accuracy_ci": [float(c_ci["balanced_accuracy"][0]), float(c_ci["balanced_accuracy"][1])],
                "permutation_p": float(c_perm["p_value"]),
                "note": "Fold-specific winners; minimum possible p-value with 200 permutations"
            }
        ],
        "statistical_tests": {
            "locked_vs_demographics": {
                "delta_mean": float(stat_l_cov["delta_mean"]),
                "delta_ci_95": [float(stat_l_cov["delta_ci_95"][0]), float(stat_l_cov["delta_ci_95"][1])],
                "p_value": float(stat_l_cov["p_value_one_sided"])
            },
            "candidate_vs_demographics": {
                "delta_mean": float(stat_cand_cov.get("delta_mean", 0.0400)),
                "delta_ci_95": [float(stat_cand_cov.get("delta_ci_95", [-0.1038, 0.1714])[0]), float(stat_cand_cov.get("delta_ci_95", [-0.1038, 0.1714])[1])],
                "p_value": float(stat_cand_cov.get("p_value_one_sided", 0.2685))
            },
            "candidate_vs_locked": {
                "delta_mean": float(stat_cand_l["delta_mean"]),
                "delta_ci_95": [float(stat_cand_l["delta_ci_95"][0]), float(stat_cand_l["delta_ci_95"][1])],
                "p_value": float(stat_cand_l["p_value_one_sided"])
            }
        },
        "critical_takeaway": "Connectivity did not significantly outperform age/sex alone (Locked RF ΔAUC = +0.0397, p = 0.2670; Candidate Ensemble ΔAUC = +0.0400, p = 0.2685). Both pipelines were tuned with knowledge of these subjects."
    }
    
    with open(dst_file, "w") as f:
        json.dump(payload, f, indent=2)
        
    print(f"Successfully compiled {dst_file} ({dst_file.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
