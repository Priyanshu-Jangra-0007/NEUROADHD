"""Generate self-contained, interactive HTML benchmark comparison report."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = BENCHMARK_ROOT / "reports"


def generate_html_report(run_id: str, results_summary: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "model_comparison.html"
    
    val_models = results_summary.get("validation_models", [])
    final_test = results_summary.get("final_test_result", {})
    best_model_name = results_summary.get("best_validation_model_name", "N/A")
    ensemble_results = results_summary.get("ensemble_results", [])
    metadata = results_summary.get("run_metadata", {})
    
    # Identify best models per validation metric
    best_by_metric = {}
    if val_models:
        metrics_keys = ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]
        for key in metrics_keys:
            best_m = max(val_models, key=lambda x: x["validation_metrics"].get(key, 0.0))
            best_by_metric[key] = {
                "model": best_m["display_name"],
                "score": best_m["validation_metrics"].get(key, 0.0)
            }

    # Generate HTML Content
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "  <title>NeuroADHD — Model Benchmark Report</title>",
        "  <style>",
        "    :root {",
        "      --bg-main: #0f172a;",
        "      --card-bg: #1e293b;",
        "      --card-border: #334155;",
        "      --text-main: #f8fafc;",
        "      --text-muted: #94a3b8;",
        "      --accent-blue: #38bdf8;",
        "      --accent-green: #4ade80;",
        "      --accent-yellow: #facc15;",
        "      --accent-red: #f87171;",
        "    }",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: var(--bg-main); color: var(--text-main); margin: 0; padding: 24px; font-size: 14px; line-height: 1.5; }",
        "    .container { max-width: 1200px; margin: 0 auto; }",
        "    .header { border-bottom: 2px solid var(--accent-blue); padding-bottom: 16px; margin-bottom: 24px; }",
        "    .header h1 { margin: 0 0 8px 0; font-size: 26px; color: var(--accent-blue); }",
        "    .header p { margin: 0; color: var(--text-muted); font-size: 13px; }",
        "    .disclaimer { background-color: rgba(248, 113, 113, 0.15); border: 1px solid var(--accent-red); color: #fca5a5; padding: 12px 16px; borderRadius: 6px; margin-bottom: 24px; font-weight: 500; }",
        "    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }",
        "    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }",
        "    .card { background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 16px; }",
        "    .card h3 { margin-top: 0; margin-bottom: 12px; font-size: 14px; color: var(--accent-blue); text-transform: uppercase; letter-spacing: 0.5px; }",
        "    .card-value { font-size: 22px; font-weight: bold; color: var(--accent-green); }",
        "    .card-label { font-size: 12px; color: var(--text-muted); }",
        "    table { width: 100%; border-collapse: collapse; margin-bottom: 24px; background-color: var(--card-bg); border-radius: 8px; overflow: hidden; }",
        "    th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--card-border); font-size: 13px; }",
        "    th { background-color: #0f172a; color: var(--accent-blue); font-weight: 600; }",
        "    tr:hover { background-color: rgba(56, 189, 248, 0.05); }",
        "    .highlight-best { background-color: rgba(74, 222, 128, 0.15); font-weight: bold; color: var(--accent-green); }",
        "    .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }",
        "    .badge-primary { background-color: rgba(56, 189, 248, 0.2); color: var(--accent-blue); }",
        "    .badge-success { background-color: rgba(74, 222, 128, 0.2); color: var(--accent-green); }",
        "    .section-title { font-size: 18px; margin-top: 32px; margin-bottom: 16px; color: var(--text-main); border-left: 4px solid var(--accent-blue); padding-left: 10px; }",
        "    .cm-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }",
        "    .cm-box { font-family: monospace; font-size: 12px; background: #0f172a; padding: 10px; border-radius: 6px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <div class='header'>",
        "      <h1>NeuroADHD — ADHD Classification Model Benchmark</h1>",
        "      <p>Reproducible Multi-Model Evaluation & Site-Disjoint Validation Dashboard | Run ID: " + str(run_id) + "</p>",
        "    </div>",
        "    <div class='disclaimer'>",
        "      <strong>⚠️ RESEARCH DISCLAIMER:</strong> This benchmarking dashboard is designed strictly for scientific machine learning research and algorithm evaluation. It is NOT a clinical diagnostic system and must NOT be used for clinical decision-making.",
        "    </div>",
        
        "    <!-- Summary Cards -->",
        "    <div class='grid-4'>",
        "      <div class='card'>",
        "        <div class='card-label'>Total Cohort</div>",
        "        <div class='card-value'>153 Subjects</div>",
        "        <div class='card-label'>50 ADHD (1) | 103 Control (0)</div>",
        "      </div>",
        "      <div class='card'>",
        "        <div class='card-label'>Data Derivative</div>",
        "        <div class='card-value'>CC200 rs-fMRI</div>",
        "        <div class='card-label'>19,900 Fisher-z → 100 Selected</div>",
        "      </div>",
        "      <div class='card'>",
        "        <div class='card-label'>Partition Strategy</div>",
        "        <div class='card-value'>Group Holdout</div>",
        "        <div class='card-label'>Train 114 | Val 19 | Test 20</div>",
        "      </div>",
        "      <div class='card'>",
        "        <div class='card-label'>Best Validation Model</div>",
        "        <div class='card-value' style='color:var(--accent-yellow); font-size:18px; margin-top:4px;'>",
        "          " + str(best_model_name),
        "        </div>",
        "        <div class='card-label'>Selected via Val ROC-AUC</div>",
        "      </div>",
        "    </div>",
        
        "    <!-- Section 1: Validation Results -->",
        "    <div class='section-title'>1. Validation Model Comparison & Hyperparameter Selection</div>",
        "    <p style='color:var(--text-muted);'>Models were trained strictly on the Training set (114 subjects) and evaluated on the held-out Validation set (19 subjects, NeuroIMAGE site). The test set remained completely untouched during this selection phase.</p>",
        "    <table>",
        "      <thead>",
        "        <tr>",
        "          <th>Model Name</th>",
        "          <th>Val Accuracy</th>",
        "          <th>Val Bal. Acc</th>",
        "          <th>Val Precision</th>",
        "          <th>Val Recall (ADHD)</th>",
        "          <th>Val Specificity (Control)</th>",
        "          <th>Val F1</th>",
        "          <th>Val ROC-AUC</th>",
        "          <th>Val PR-AUC</th>",
        "          <th>Train Time (s)</th>",
        "        </tr>",
        "      </thead>",
        "      <tbody>"
    ]
    
    for m in val_models:
        v = m["validation_metrics"]
        dname = m["display_name"]
        is_best = (dname == best_model_name or m["model_name"] == best_model_name)
        row_cls = "class='highlight-best'" if is_best else ""
        html_lines.append(f"        <tr {row_cls}>")
        html_lines.append(f"          <td><strong>{dname}</strong> " + ("<span class='badge badge-success'>SELECTED</span>" if is_best else "") + "</td>")
        html_lines.append(f"          <td>{v['accuracy']:.4f}</td>")
        html_lines.append(f"          <td>{v['balanced_accuracy']:.4f}</td>")
        html_lines.append(f"          <td>{v['precision']:.4f}</td>")
        html_lines.append(f"          <td>{v['recall_sensitivity']:.4f}</td>")
        html_lines.append(f"          <td>{v['specificity']:.4f}</td>")
        html_lines.append(f"          <td>{v['f1_score']:.4f}</td>")
        html_lines.append(f"          <td><strong>{v['roc_auc']:.4f}</strong></td>")
        html_lines.append(f"          <td>{v['pr_auc']:.4f}</td>")
        html_lines.append(f"          <td>{m['timing_seconds']['total_execution_time']:.2f}s</td>")
        html_lines.append("        </tr>")
        
    html_lines.extend([
        "      </tbody>",
        "    </table>",
        
        "    <!-- Section 2: Best Models by Metric -->",
        "    <div class='section-title'>2. Best Validation Performers by Metric</div>",
        "    <div class='grid-4'>"
    ])
    
    for metric_key, label in [("roc_auc", "Best ROC-AUC"), ("accuracy", "Best Accuracy"), ("f1_score", "Best F1 Score"), ("balanced_accuracy", "Best Balanced Acc")]:
        b_info = best_by_metric.get(metric_key, {"model": "N/A", "score": 0.0})
        html_lines.append(f"      <div class='card'>")
        html_lines.append(f"        <div class='card-label'>{label}</div>")
        html_lines.append(f"        <div class='card-value' style='font-size:16px; color:var(--accent-blue); margin-top:4px;'>{b_info['model']}</div>")
        html_lines.append(f"        <div class='card-label' style='margin-top:4px; font-weight:bold;'>Score: {b_info['score']:.4f}</div>")
        html_lines.append(f"      </div>")
        
    html_lines.extend([
        "    </div>",
        
        "    <!-- Section 3: Ensemble Phase -->",
        "    <div class='section-title'>3. Optional Ensemble / Hybrid Models Evaluation</div>",
        "    <p style='color:var(--text-muted);'>Ensembles combine predictions from top-performing base classifiers. Stacking uses out-of-fold predictions on training data to train the meta-estimator without leakage.</p>",
        "    <table>",
        "      <thead>",
        "        <tr>",
        "          <th>Ensemble Strategy</th>",
        "          <th>Base Models</th>",
        "          <th>Val Accuracy</th>",
        "          <th>Val F1</th>",
        "          <th>Val ROC-AUC</th>",
        "          <th>Val PR-AUC</th>",
        "        </tr>",
        "      </thead>",
        "      <tbody>"
    ])
    
    for ens in ensemble_results:
        ev = ens["validation_metrics"]
        html_lines.append("        <tr>")
        html_lines.append(f"          <td><strong>{ens['display_name']}</strong></td>")
        html_lines.append(f"          <td>{', '.join(ens.get('base_models', []))}</td>")
        html_lines.append(f"          <td>{ev['accuracy']:.4f}</td>")
        html_lines.append(f"          <td>{ev['f1_score']:.4f}</td>")
        html_lines.append(f"          <td><strong>{ev['roc_auc']:.4f}</strong></td>")
        html_lines.append(f"          <td>{ev['pr_auc']:.4f}</td>")
        html_lines.append("        </tr>")
        
    html_lines.extend([
        "      </tbody>",
        "    </table>",
        
        "    <!-- Section 4: Final Test Evaluation -->",
        "    <div class='section-title'>4. Final Test Results (Strictly Locked Peking_2 Test Site)</div>",
        "    <p style='color:var(--text-muted);'>After model selection on Validation, the single best model was refit on Train + Validation data and evaluated <strong>ONCE</strong> on the held-out Peking_2 Test set (20 subjects: 6 ADHD, 14 Control).</p>"
    ])
    
    if final_test:
        ft_m = final_test.get("metrics", {})
        cm = ft_m.get("confusion_matrix", {})
        html_lines.extend([
            "    <div class='card' style='border:2px solid var(--accent-green); margin-bottom:24px;'>",
            "      <h3 style='color:var(--accent-green);'>Final Test Summary — " + str(final_test.get("model_name", "")) + "</h3>",
            "      <div class='grid-4' style='margin-bottom:0;'>",
            "        <div><div class='card-label'>Test Accuracy</div><div class='card-value'>" + f"{ft_m.get('accuracy', 0):.4f}" + "</div></div>",
            "        <div><div class='card-label'>Test Balanced Acc</div><div class='card-value'>" + f"{ft_m.get('balanced_accuracy', 0):.4f}" + "</div></div>",
            "        <div><div class='card-label'>Test Sensitivity (ADHD)</div><div class='card-value'>" + f"{ft_m.get('recall_sensitivity', 0):.4f}" + "</div></div>",
            "        <div><div class='card-label'>Test Specificity (Control)</div><div class='card-value'>" + f"{ft_m.get('specificity', 0):.4f}" + "</div></div>",
            "        <div><div class='card-label'>Test Precision</div><div class='card-value'>" + f"{ft_m.get('precision', 0):.4f}" + "</div></div>",
            "        <div><div class='card-label'>Test F1 Score</div><div class='card-value'>" + f"{ft_m.get('f1_score', 0):.4f}" + "</div></div>",
            "        <div><div class='card-label'>Test ROC-AUC</div><div class='card-value' style='color:var(--accent-yellow);'>" + f"{ft_m.get('roc_auc', 0):.4f}" + "</div></div>",
            "        <div><div class='card-label'>Test PR-AUC</div><div class='card-value'>" + f"{ft_m.get('pr_auc', 0):.4f}" + "</div></div>",
            "      </div>",
            "      <div style='margin-top:16px; padding-top:12px; border-top:1px solid var(--card-border);'>",
            "        <strong>Final Test Confusion Matrix:</strong> ",
            "        TP (ADHD correct) = " + str(cm.get("tp", 0)) + " | ",
            "        TN (Control correct) = " + str(cm.get("tn", 0)) + " | ",
            "        FP (False ADHD) = " + str(cm.get("fp", 0)) + " | ",
            "        FN (Missed ADHD) = " + str(cm.get("fn", 0)),
            "      </div>",
            "    </div>"
        ])

    html_lines.extend([
        "    <!-- Section 5: Reproducibility & Environment Metadata -->",
        "    <div class='section-title'>5. Reproducibility & Metadata</div>",
        "    <div class='card'>",
        "      <div style='display:grid; grid-template-columns: 1fr 1fr; gap:12px; font-size:12px;'>",
        "        <div><strong>Run ID:</strong> " + str(run_id) + "</div>",
        "        <div><strong>Random Seed:</strong> 42</div>",
        "        <div><strong>Python Executable:</strong> " + str(metadata.get("python_version", sys.version.split()[0])) + "</div>",
        "        <div><strong>Operating System:</strong> " + str(metadata.get("os", sys.platform)) + "</div>",
        "        <div><strong>Primary Feature Table:</strong> NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv</div>",
        "        <div><strong>Split Manifests:</strong> NeuroADHD/data/adhd200/processed/splits/</div>",
        "      </div>",
        "    </div>",
        
        "  </div>",
        "</body>",
        "</html>"
    ])
    
    report_file.write_text("\n".join(html_lines), encoding="utf-8")
    print(f"HTML comparison report successfully generated: {report_file}")
    return report_file


if __name__ == "__main__":
    pass
