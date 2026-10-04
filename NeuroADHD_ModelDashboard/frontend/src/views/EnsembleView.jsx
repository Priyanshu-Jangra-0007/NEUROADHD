import React from "react";
import { Cpu, CheckCircle, TrendingUp, AlertCircle } from "lucide-react";

export function EnsembleView({ results }) {
  const ensModels = results?.ensemble_results || [];
  const valModels = results?.validation_models || [];

  const topIndividual = valModels.length ? [...valModels].sort((a, b) => (b.validation_metrics?.roc_auc || 0) - (a.validation_metrics?.roc_auc || 0))[0] : null;
  const topIndRoc = topIndividual?.validation_metrics?.roc_auc || 0;

  return (
    <div className="view-container">
      <h2 style={{ fontSize: "20px", fontWeight: "700", marginBottom: "6px" }}>Ensemble & Hybrid Model Benchmarks</h2>
      <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "24px" }}>Evaluate Soft Voting, Hard Voting, and Stacking Meta-Classifiers against the best individual model.</p>

      {topIndividual && (
        <div className="card" style={{ marginBottom: "24px", background: "var(--bg-subtle)" }}>
          <div className="card-title">Top Individual Reference Model</div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <h3 style={{ fontSize: "18px", fontWeight: "700" }}>{topIndividual.display_name}</h3>
              <p style={{ fontSize: "12px", color: "var(--text-muted)" }}>Best individual baseline classifier</p>
            </div>
            <div style={{ fontSize: "20px", fontWeight: "700", color: "var(--brand-blue)" }}>
              {(topIndRoc * 100).toFixed(2)}% ROC-AUC
            </div>
          </div>
        </div>
      )}

      {/* Ensemble Comparison Table */}
      <div className="table-container">
        <div className="table-header-bar">
          <h3>Ensemble Strategies & Performance Summary</h3>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Ensemble Strategy</th>
              <th>Base Classifiers Included</th>
              <th>Val Accuracy</th>
              <th>Val F1</th>
              <th>Val ROC-AUC</th>
              <th>vs. Best Individual</th>
            </tr>
          </thead>
          <tbody>
            {ensModels.map((ens, idx) => {
              const vm = ens.validation_metrics || {};
              const diff = (vm.roc_auc || 0) - topIndRoc;
              const improved = diff > 0.0001;
              return (
                <tr key={idx}>
                  <td><strong>{ens.display_name}</strong></td>
                  <td>{ens.base_models ? ens.base_models.join(", ") : "All Base Models"}</td>
                  <td>{(vm.accuracy * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.f1_score * 100 || 0).toFixed(2)}%</td>
                  <td style={{ fontWeight: "700", color: "var(--brand-blue)" }}>{(vm.roc_auc * 100 || 0).toFixed(2)}%</td>
                  <td>
                    {improved ? (
                      <span style={{ color: "var(--accent-green)", fontWeight: "600" }}>+{(diff * 100).toFixed(2)}% (Improved)</span>
                    ) : (
                      <span style={{ color: "var(--text-muted)" }}>{(diff * 100).toFixed(2)}% (No Gain)</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
