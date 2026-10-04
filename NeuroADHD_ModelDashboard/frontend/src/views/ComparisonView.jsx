import React from "react";
import { Trophy, BarChart2 } from "lucide-react";

export function ComparisonView({ results }) {
  const valModels = results?.validation_models || [];
  const ensModels = results?.ensemble_results || [];
  const allModels = [...valModels, ...ensModels];

  const getBestForMetric = (metric) => {
    if (!allModels.length) return { name: "N/A", val: 0 };
    const sorted = [...allModels].sort((a, b) => (b.validation_metrics?.[metric] || 0) - (a.validation_metrics?.[metric] || 0));
    return {
      name: sorted[0].display_name || sorted[0].model_name,
      val: sorted[0].validation_metrics?.[metric] || 0
    };
  };

  const bestRoc = getBestForMetric("roc_auc");
  const bestAcc = getBestForMetric("accuracy");
  const bestF1 = getBestForMetric("f1_score");
  const bestBalAcc = getBestForMetric("balanced_accuracy");

  return (
    <div className="view-container">
      <h2 style={{ fontSize: "20px", fontWeight: "700", marginBottom: "6px" }}>Model Performance Comparison</h2>
      <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "24px" }}>Side-by-side evaluation of all classifiers on the validation holdout set.</p>

      {/* Best Model Callout Grid */}
      <h3 style={{ fontSize: "13px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "12px", letterSpacing: "0.5px" }}>Best Performers by Metric</h3>
      <div className="grid-cards">
        <div className="card" style={{ borderLeft: "4px solid var(--brand-blue)" }}>
          <div className="card-title">Best Validation ROC-AUC</div>
          <div className="card-value" style={{ fontSize: "18px", color: "var(--brand-blue)" }}>{bestRoc.name}</div>
          <div className="card-subtext">Score: {(bestRoc.val * 100).toFixed(2)}%</div>
        </div>
        <div className="card" style={{ borderLeft: "4px solid var(--accent-green)" }}>
          <div className="card-title">Best Validation Accuracy</div>
          <div className="card-value" style={{ fontSize: "18px", color: "var(--accent-green)" }}>{bestAcc.name}</div>
          <div className="card-subtext">Score: {(bestAcc.val * 100).toFixed(2)}%</div>
        </div>
        <div className="card" style={{ borderLeft: "4px solid var(--accent-amber)" }}>
          <div className="card-title">Best Validation F1 Score</div>
          <div className="card-value" style={{ fontSize: "18px", color: "var(--accent-amber)" }}>{bestF1.name}</div>
          <div className="card-subtext">Score: {(bestF1.val * 100).toFixed(2)}%</div>
        </div>
        <div className="card" style={{ borderLeft: "4px solid #8b5cf6" }}>
          <div className="card-title">Best Balanced Accuracy</div>
          <div className="card-value" style={{ fontSize: "18px", color: "#8b5cf6" }}>{bestBalAcc.name}</div>
          <div className="card-subtext">Score: {(bestBalAcc.val * 100).toFixed(2)}%</div>
        </div>
      </div>

      {/* Side-by-Side Comparison Table */}
      <div className="table-container">
        <div className="table-header-bar">
          <h3>Full Validation Benchmark Table</h3>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Model Name</th>
              <th>Val Accuracy</th>
              <th>Val Bal. Acc</th>
              <th>Val Precision</th>
              <th>Val Recall</th>
              <th>Val Specificity</th>
              <th>Val F1</th>
              <th>Val ROC-AUC</th>
              <th>Val PR-AUC</th>
            </tr>
          </thead>
          <tbody>
            {allModels.map((m, idx) => {
              const vm = m.validation_metrics || {};
              return (
                <tr key={idx}>
                  <td><strong>{m.display_name || m.model_name}</strong></td>
                  <td>{(vm.accuracy * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.balanced_accuracy * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.precision * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.recall_sensitivity * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.specificity * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.f1_score * 100 || 0).toFixed(2)}%</td>
                  <td style={{ fontWeight: "700", color: "var(--brand-blue)" }}>{(vm.roc_auc * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.pr_auc * 100 || 0).toFixed(2)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
