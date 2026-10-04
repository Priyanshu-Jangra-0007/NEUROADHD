import React from "react";
import { AlertCircle } from "lucide-react";

export function HybridComparisonView({ data }) {
  const rfStats = data.rf_robustness.summary.all_8_repetitions;
  const hybridStats = data.hybrid_comparison.summary;

  const comparisonRows = [
    {
      name: "Balanced Random Forest (Single Reference)",
      type: "Standalone Model",
      protocol: "8-Repetition 5-Fold CV",
      roc_auc: `${rfStats.roc_auc.mean.toFixed(4)} ± ${rfStats.roc_auc.std.toFixed(4)}`,
      bal_acc: `${rfStats.balanced_accuracy.mean.toFixed(4)} ± ${rfStats.balanced_accuracy.std.toFixed(4)}`,
      sensitivity: `${rfStats.recall_sensitivity.mean.toFixed(4)} ± ${rfStats.recall_sensitivity.std.toFixed(4)}`,
      f1: `${rfStats.f1_score.mean.toFixed(4)} ± ${rfStats.f1_score.std.toFixed(4)}`,
      isBest: true
    },
    {
      name: "Soft Voting (0.6 RF / 0.4 XGBoost)",
      type: "Ensemble Hybrid",
      protocol: "5-Repetition 5-Fold CV",
      roc_auc: `${hybridStats.soft_voting.roc_auc.mean.toFixed(4)} ± ${hybridStats.soft_voting.roc_auc.std.toFixed(4)}`,
      bal_acc: `${hybridStats.soft_voting.balanced_accuracy.mean.toFixed(4)} ± ${hybridStats.soft_voting.balanced_accuracy.std.toFixed(4)}`,
      sensitivity: `${hybridStats.soft_voting.recall_sensitivity.mean.toFixed(4)} ± ${hybridStats.soft_voting.recall_sensitivity.std.toFixed(4)}`,
      f1: `${hybridStats.soft_voting.f1_score.mean.toFixed(4)} ± ${hybridStats.soft_voting.f1_score.std.toFixed(4)}`,
      isBest: false
    },
    {
      name: "Hard Voting (RF + XGBoost)",
      type: "Ensemble Hybrid",
      protocol: "5-Repetition 5-Fold CV",
      roc_auc: `${hybridStats.hard_voting.roc_auc.mean.toFixed(4)} ± ${hybridStats.hard_voting.roc_auc.std.toFixed(4)}`,
      bal_acc: `${hybridStats.hard_voting.balanced_accuracy.mean.toFixed(4)} ± ${hybridStats.hard_voting.balanced_accuracy.std.toFixed(4)}`,
      sensitivity: `${hybridStats.hard_voting.recall_sensitivity.mean.toFixed(4)} ± ${hybridStats.hard_voting.recall_sensitivity.std.toFixed(4)}`,
      f1: `${hybridStats.hard_voting.f1_score.mean.toFixed(4)} ± ${hybridStats.hard_voting.f1_score.std.toFixed(4)}`,
      isBest: false
    },
    {
      name: "Stacking (Balanced Logistic Regression Meta)",
      type: "Ensemble Hybrid",
      protocol: "5-Repetition 5-Fold CV",
      roc_auc: `${hybridStats.stacking.roc_auc.mean.toFixed(4)} ± ${hybridStats.stacking.roc_auc.std.toFixed(4)}`,
      bal_acc: `${hybridStats.stacking.balanced_accuracy.mean.toFixed(4)} ± ${hybridStats.stacking.balanced_accuracy.std.toFixed(4)}`,
      sensitivity: `${hybridStats.stacking.recall_sensitivity.mean.toFixed(4)} ± ${hybridStats.stacking.recall_sensitivity.std.toFixed(4)}`,
      f1: `${hybridStats.stacking.f1_score.mean.toFixed(4)} ± ${hybridStats.stacking.f1_score.std.toFixed(4)}`,
      isBest: false
    }
  ];

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: "800", color: "var(--text-primary)" }}>Single Model vs. Hybrid Architecture Comparison</h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          Empirical comparison between the frozen Balanced Random Forest and three ensemble architectures under identical repeated cross-validation protocols.
        </p>
      </div>

      <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "20px", marginBottom: "20px" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-color)", textAlign: "left", color: "var(--text-muted)" }}>
              <th style={{ padding: "10px" }}>Model Architecture</th>
              <th style={{ padding: "10px" }}>Type</th>
              <th style={{ padding: "10px" }}>Protocol</th>
              <th style={{ padding: "10px" }}>ROC-AUC (Mean ± SD)</th>
              <th style={{ padding: "10px" }}>Balanced Accuracy</th>
              <th style={{ padding: "10px" }}>Sensitivity</th>
              <th style={{ padding: "10px" }}>F1 Score</th>
            </tr>
          </thead>
          <tbody>
            {comparisonRows.map((r, i) => (
              <tr key={i} style={{ borderBottom: "1px solid var(--border-color)", background: r.isBest ? "var(--bg-subtle)" : "transparent" }}>
                <td style={{ padding: "10px", fontWeight: r.isBest ? "700" : "500", color: r.isBest ? "var(--brand-blue)" : "var(--text-primary)" }}>
                  {r.name} {r.isBest && <span style={{ fontSize: "0.7rem", background: "var(--accent-green-bg)", color: "var(--accent-green)", padding: "2px 6px", borderRadius: "4px", marginLeft: "6px" }}>SELECTED</span>}
                </td>
                <td style={{ padding: "10px", color: "var(--text-muted)", fontSize: "0.75rem" }}>{r.type}</td>
                <td style={{ padding: "10px", color: "var(--text-muted)", fontSize: "0.75rem" }}>{r.protocol}</td>
                <td style={{ padding: "10px", fontWeight: "700" }}>{r.roc_auc}</td>
                <td style={{ padding: "10px" }}>{r.bal_acc}</td>
                <td style={{ padding: "10px" }}>{r.sensitivity}</td>
                <td style={{ padding: "10px" }}>{r.f1}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ background: "#f8fafc", border: "1px solid var(--border-color)", borderRadius: "8px", padding: "16px", display: "flex", alignItems: "flex-start", gap: "12px" }}>
        <AlertCircle size={20} color="var(--brand-blue)" style={{ flexShrink: 0, marginTop: "2px" }} />
        <div>
          <h4 style={{ fontSize: "0.9rem", fontWeight: "700", color: "var(--text-primary)", marginBottom: "4px" }}>Scientific Conclusion</h4>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: "1.5" }}>
            <strong>Hybrid models did not demonstrate a consistent improvement over the Balanced Random Forest.</strong> Across repeated cross-validation splits, the standalone Balanced Random Forest achieved higher average ROC-AUC (0.6045 vs 0.5926 for voting and 0.5555 for stacking) with superior parsimony and stability. Consequently, the single Balanced Random Forest was finalized.
          </p>
        </div>
      </div>
    </div>
  );
}
