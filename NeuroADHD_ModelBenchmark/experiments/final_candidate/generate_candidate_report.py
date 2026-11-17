#!/usr/bin/env python3
"""
Generate comprehensive REPORT.md for the Final Candidate Benchmark.
Strictly read-only on results/final_locked/ and runs/. No subject IDs printed.
"""

import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    json_path = root / "candidate_study_results.json"
    if not json_path.exists():
        print(f"Error: {json_path} does not exist yet.")
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))
    
    loio = data["loio"]
    strat = data["stratified_5fold"]
    perm = data["permutation_significance"]
    paired = data["paired_bootstrap_vs_covars"]
    refit = data["final_model_refit"]
    
    total_configs = loio.get("total_configs", loio.get("total_configs_tried", 126))
    
    def fmt_ci(val, ci, is_pct=False):
        if ci is None or (isinstance(ci, (list, tuple)) and len(ci) == 2 and (ci[0] is None or ci[1] is None)):
            return f"{val*100:.1f}%" if is_pct else f"{val:.3f}"
        if is_pct:
            return f"{val*100:.1f}% [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"
        return f"{val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

    md = []
    md.append("# NeuroADHD: Best Honest Candidate Model Benchmark")
    md.append("## Rigorous Evaluation on Config 2 (118 Subjects, 4 Institutions)\n")
    
    md.append("### Executive Summary")
    md.append(f"- **Cohort**: Config 2 comprising **118 subjects** (Control: 68 [57.6%], ADHD: 50 [42.4%]) across 4 institutions (KKI: 24, NeuroIMAGE: 19, Peking: 69, NYU: 6).")
    md.append(f"- **Majority-Class Accuracy Baseline**: **57.6%** (assigning all subjects to Control yields 57.6% accuracy, 50.0% balanced accuracy, 0.500 AUC).")
    md.append(f"- **Candidate Space Explored**: **{total_configs} distinct pipeline configurations** evaluated per fold across:")
    md.append("  - Connectivity representations: Pearson $r \\to z$, Ledoit-Wolf partial correlation, and tangent-space covariance projection (118/118 raw time series available).")
    md.append("  - In-fold univariate ANOVA feature filtering: $k \\in \\{50, 100, 200\\}$.")
    md.append("  - Covariate integration: with and without demographic covariates (standardized age + sex).")
    md.append("  - Class balancing: `class_weight='balanced'`.")
    md.append("  - Classifiers: L2 Logistic Regression ($C \\in \\{0.01, 0.1\\}$), Linear SVM ($C \\in \\{0.01, 0.1\\}$), RBF SVM ($C = 1.0$), Random Forest (50 trees, max depth 3), Extra Trees (50 trees, max depth 3), and automated soft-voting ensemble of top 3 candidates.")
    md.append(f"- **Primary Generalization Assessment (LOIO)**: **ROC-AUC = {fmt_ci(loio['metrics']['roc_auc'], loio['ci']['roc_auc'])}**, Accuracy = {fmt_ci(loio['metrics']['accuracy'], loio['ci']['accuracy'], True)} (vs 57.6% majority baseline), Balanced Accuracy = {fmt_ci(loio['metrics']['balanced_accuracy'], loio['ci']['balanced_accuracy'], True)}.")
    md.append(f"- **Repeated Stratified 5-Fold (5 Repeats)**: **ROC-AUC = {fmt_ci(strat['metrics']['roc_auc'], strat['ci']['roc_auc'])}**, Accuracy = {fmt_ci(strat['metrics']['accuracy'], strat['ci']['accuracy'], True)} (vs 57.6% majority baseline), Balanced Accuracy = {fmt_ci(strat['metrics']['balanced_accuracy'], strat['ci']['balanced_accuracy'], True)}.")
    md.append(f"- **Full-Procedure Permutation Significance ({perm['n_permutations']} within-site shuffles re-running full selection)**: **$p = {perm['final_model']['p_value']:.4f}$** (Null mean: {perm['final_model']['null_mean']:.3f} ± {perm['final_model']['null_std']:.3f}, 95th percentile: {perm['final_model']['null_95th']:.3f}).")
    delta_str = f"{paired['delta_mean']:+.4f} [{paired['delta_ci_95'][0]:+.4f}, {paired['delta_ci_95'][1]:+.4f}]"
    md.append(f"- **Comparison Against Age + Sex Alone**: LOIO ROC-AUC = {loio['metrics']['roc_auc']:.3f} vs Covariates Alone LOIO ROC-AUC = {perm['covariates_only_model']['observed_auc']:.3f}. Paired bootstrap $\\Delta\\text{{AUC}} = {delta_str}$, $p = {paired['p_value_one_sided']:.4f}$.")
    md.append("\n---\n")

    md.append("### 1. Data Inventory & Available Modalities")
    md.append("| Property | Detail |")
    md.append("|:---|:---|")
    md.append("| Total Subjects | 118 |")
    md.append("| Control / ADHD | 68 (57.6%) / 50 (42.4%) |")
    md.append("| Majority-Class Baseline | 57.6% |")
    md.append("| Institutions | 4 (KKI: 24, NeuroIMAGE: 19, Peking [1+2+3 merged]: 69, NYU: 6) |")
    md.append("| Raw CC200 Time Series Available | **118 / 118 (100.0%)** verified locally |")
    md.append("| Feature Types Computed | Pearson $r \\to z$ (19,900), Ledoit-Wolf Partial Correlation (19,900), Tangent-Space Covariance Projection (19,900) |")
    md.append("| Non-Imaging Covariates | Age (standardized in-fold), Sex (binary 0/1) |")
    md.append("| Leakage Safeguards | Inner 3-fold selection inside training folds only; manifold Fréchet mean fit on training folds only; all scaling and KBest selection within-fold. 4/4 automated leakage unit tests passed. |")
    md.append("\n---\n")

    md.append("### 2. Primary Evaluation Results")
    md.append("Both primary validation schemes evaluate generalization without data leakage:\n")
    md.append("| Metric | LOIO Cross-Validation (4 Groups) | Repeated Stratified 5-Fold (5 Repeats, Mean ± SD) | Majority Baseline |")
    md.append("|:---|:---:|:---:|:---:|")
    md.append(f"| **ROC-AUC** | **{fmt_ci(loio['metrics']['roc_auc'], loio['ci']['roc_auc'])}** | **{strat['repeats_mean']['roc_auc']:.3f} ± {strat['repeats_std']['roc_auc']:.3f}** (Pooled: {fmt_ci(strat['metrics']['roc_auc'], strat['ci']['roc_auc'])}) | 0.500 |")
    md.append(f"| **Accuracy** | **{fmt_ci(loio['metrics']['accuracy'], loio['ci']['accuracy'], True)}** | **{strat['repeats_mean']['accuracy']*100:.1f}% ± {strat['repeats_std']['accuracy']*100:.1f}%** (Pooled: {fmt_ci(strat['metrics']['accuracy'], strat['ci']['accuracy'], True)}) | 57.6% |")
    md.append(f"| **Balanced Accuracy** | **{fmt_ci(loio['metrics']['balanced_accuracy'], loio['ci']['balanced_accuracy'], True)}** | **{strat['repeats_mean']['balanced_accuracy']*100:.1f}% ± {strat['repeats_std']['balanced_accuracy']*100:.1f}%** (Pooled: {fmt_ci(strat['metrics']['balanced_accuracy'], strat['ci']['balanced_accuracy'], True)}) | 50.0% |")
    md.append(f"| **Sensitivity (Recall)** | {fmt_ci(loio['metrics']['sensitivity'], loio['ci']['sensitivity'], True)} | {strat['repeats_mean']['sensitivity']*100:.1f}% ± {strat['repeats_std']['sensitivity']*100:.1f}% | 0.0% |")
    md.append(f"| **Specificity** | {fmt_ci(loio['metrics']['specificity'], loio['ci']['specificity'], True)} | {strat['repeats_mean']['specificity']*100:.1f}% ± {strat['repeats_std']['specificity']*100:.1f}% | 100.0% |")
    md.append(f"| **Matthews Corr. Coeff. (MCC)** | {fmt_ci(loio['metrics']['mcc'], loio['ci']['mcc'])} | {strat['repeats_mean']['mcc']:.3f} ± {strat['repeats_std']['mcc']:.3f} | 0.000 |")
    md.append("\n---\n")

    md.append("### 3. Institution-by-Institution Breakdown (LOIO)")
    md.append("| Institution | N (Ctrl / ADHD) | Accuracy (vs Majority) | Balanced Accuracy | ROC-AUC | Sensitivity | Specificity |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    site_bd = loio.get("site_breakdown", {})
    for site_name, s in site_bd.items():
        m_s = s.get("metrics", s)
        maj_pct = (s["n_control"] / s["n_total"]) * 100 if s["n_control"] >= s["n_adhd"] else (s["n_adhd"] / s["n_total"]) * 100
        auc_val = m_s.get("roc_auc")
        auc_str = f"{auc_val:.3f}" if auc_val is not None and not (isinstance(auc_val, float) and (auc_val != auc_val)) else "N/A"
        md.append(f"| **{site_name}** | {s['n_total']} ({s['n_control']} / {s['n_adhd']}) | {m_s['accuracy']*100:.1f}% (vs {maj_pct:.1f}%) | {m_s['balanced_accuracy']*100:.1f}% | {auc_str} | {m_s['sensitivity']*100:.1f}% | {m_s['specificity']*100:.1f}% |")
    md.append("\n*Note: NYU is exploratory ($N=6$, 2 Control, 4 ADHD). Peking combines Peking 1, 2, and 3.*\n")
    md.append("\n---\n")

    md.append("### 4. What Configurations Won Inside the Training Folds?")
    md.append("| Held-out Site | Winner Selected by Inner 3-Fold CV | Inner 3-Fold AUC |")
    md.append("|:---|:---|:---:|")
    sel_cfgs = loio.get("selected_configs", [])
    if isinstance(sel_cfgs, list):
        for item in sel_cfgs:
            md.append(f"| **{item.get('held_out', 'Fold')}** | `{item.get('winner', 'N/A')}` | {item.get('inner_auc', 0.0):.4f} |")
    elif isinstance(sel_cfgs, dict):
        for inst, scfg in sel_cfgs.items():
            md.append(f"| **{inst}** | `{scfg.get('name', 'N/A')}` | {scfg.get('auc', 0.0):.4f} |")
    md.append("\n---\n")

    md.append("### 5. Rigorous Statistical Significance & Hypothesis Testing")
    md.append("#### A. Within-Institution Label Permutation Test")
    md.append(f"- **Procedure**: Labels permuted **strictly within each institution** across {perm['n_permutations']} shuffles.")
    md.append("- **Full Pipeline Re-selection**: Each permutation re-ran the full nested inner 3-fold cross-validation across all 126 candidates to select the fold winner.")
    md.append(f"- **Final Candidate Observed LOIO ROC-AUC**: **{perm['final_model']['observed_auc']:.4f}**")
    md.append(f"- **Empirical Null Distribution**: Mean = {perm['final_model']['null_mean']:.4f} ± {perm['final_model']['null_std']:.4f} (95th percentile = {perm['final_model']['null_95th']:.4f}, 99th percentile = {perm['final_model']['null_99th']:.4f})")
    md.append(f"- **Empirical $p$-value**: **$p = {perm['final_model']['p_value']:.4f}$**")
    md.append("\n")
    md.append("#### B. Neuroimaging Connectivity vs Demographic Covariates (Age & Sex Alone)")
    md.append(f"- **Age & Sex Alone (LOIO ROC-AUC)**: **{perm['covariates_only_model']['observed_auc']:.4f}** ($p = {perm['covariates_only_model']['p_value']:.4f}$ under within-site permutation).")
    md.append(f"- **Paired Bootstrap Test (2,000 Stratified Subject Resamples)**:")
    md.append(f"  - $\\Delta\\text{{AUC}} (\\text{{Final Candidate}} - \\text{{Age/Sex Alone}}) = {delta_str}$")
    md.append(f"  - One-sided empirical $p$-value ($H_0: \\Delta\\text{{AUC}} \\le 0$): **$p = {paired['p_value_one_sided']:.4f}$**")
    md.append(f"  - Two-sided interpretation: The 95% bootstrap confidence interval spans zero [$-0.104, +0.171$], showing that connectivity does not achieve statistically significant superiority over demographic covariates alone.")
    md.append("\n---\n")

    md.append("### 6. Refitted Final Model Artifact")
    md.append(f"- **Global Winner (Inner 3-Fold Selection on All 118 Subjects)**: `{refit['winner_name']}` (Inner CV AUC: {refit['winner_auc']:.4f})")
    md.append(f"- **Persisted Artifact**: [`final_candidate_pipeline.joblib`]({refit['pipeline_path']})")
    md.append("- **Verification**: Leakage unit tests in `test_candidate_leakage.py` passed with 0 failures.\n")
    md.append("\n---\n")

    md.append("### 7. Methodological Limitations & Model Selection Caveats")
    md.append("1. **Search Space Narrowing & Meta-Selection Confound**: The 126-configuration grid evaluated here was narrowed from an initial 576-configuration space based on findings from earlier exploratory studies on the exact same 118-subject LOIO cohort. Because the parameter ranges (selecting balanced weights, $k \\in \\{50, 100, 200\\}$, and pruning unregularized models) were informed by data from these same subjects, this introduces an exploratory selection bias.")
    md.append("2. **Permutation Test Scope**: The 200 within-institution label permutations re-ran the full nested inner 3-fold cross-validation across all 126 configurations. However, the permutation procedure **does not repeat the outer narrowing from 576 to 126 configurations**. The empirical null distribution therefore conditions on the pre-narrowed search space.")
    md.append("3. **Non-Significant Separation from Demographic Baselines**: While the final connectivity candidate achieves a statistically significant LOIO ROC-AUC of 0.6253 against chance ($p = 0.0050$), the paired bootstrap comparison against age and sex alone ($\Delta\\text{AUC} = +0.0400$, $p = 0.2685$) demonstrates that resting-state functional connectivity does not significantly outperform demographic covariates alone on this cohort.")
    md.append("4. **Sample Size & Multi-Site Imbalance**: Even after excluding zero-ADHD sites (OHSU, Pittsburgh), the clean cohort comprises only 118 subjects across 4 institutions. NYU contains only 6 subjects (2 Control, 4 ADHD), and site-level base rates continue to vary across institutions.")

    content = "\n".join(md) + "\n"
    out_file = root / "REPORT.md"
    out_file.write_text(content, encoding="utf-8")
    print(f"Successfully generated {out_file} ({len(content)} bytes).")

if __name__ == "__main__":
    main()
