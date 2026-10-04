import React from "react";
import { Activity, Lock } from "lucide-react";

export function PerformanceView({ data }) {
  const rfStats = data.rf_robustness.summary.all_8_repetitions;
  const reps = data.rf_robustness.repetitions;
  const finalTest = data.final_test_results.metrics;
  const cm = data.final_confusion_matrix;

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: "800", color: "var(--text-primary)" }}>Model Performance Evaluation</h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          Rigorous dual reporting: Multi-seed repeated cross-validation robustness across the development cohort and one-time locked final evaluation on Peking_2.
        </p>
      </div>

      {/* BLOCK A: REPEATED CV ROBUSTNESS */}
      <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "20px", marginBottom: "24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Activity size={20} color="var(--brand-blue)" />
            <h3 style={{ fontSize: "1.1rem", fontWeight: "800", color: "var(--text-primary)" }}>
              Block A: Repeated Cross-Validation Robustness (8 Repetitions)
            </h3>
          </div>
          <span style={{ fontSize: "0.75rem", background: "var(--brand-blue-bg)", color: "var(--brand-blue)", padding: "4px 10px", borderRadius: "9999px", fontWeight: "700" }}>
            5-Fold CV × Seeds 42–49 (N=133)
          </span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "12px", marginBottom: "16px" }}>
          <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "8px" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Mean ROC-AUC</div>
            <div style={{ fontSize: "1.3rem", fontWeight: "800", color: "var(--brand-blue)" }}>{rfStats.roc_auc.mean.toFixed(4)}</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>± {rfStats.roc_auc.std.toFixed(4)} [{rfStats.roc_auc.min.toFixed(4)}–{rfStats.roc_auc.max.toFixed(4)}]</div>
          </div>
          <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "8px" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Mean Accuracy</div>
            <div style={{ fontSize: "1.3rem", fontWeight: "800" }}>{rfStats.accuracy.mean.toFixed(4)}</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>± {rfStats.accuracy.std.toFixed(4)}</div>
          </div>
          <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "8px" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Mean Balanced Acc</div>
            <div style={{ fontSize: "1.3rem", fontWeight: "800" }}>{rfStats.balanced_accuracy.mean.toFixed(4)}</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>± {rfStats.balanced_accuracy.std.toFixed(4)}</div>
          </div>
          <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "8px" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Mean Sensitivity</div>
            <div style={{ fontSize: "1.3rem", fontWeight: "800" }}>{rfStats.recall_sensitivity.mean.toFixed(4)}</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>± {rfStats.recall_sensitivity.std.toFixed(4)}</div>
          </div>
          <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "8px" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Mean Specificity</div>
            <div style={{ fontSize: "1.3rem", fontWeight: "800" }}>{rfStats.specificity.mean.toFixed(4)}</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>± {rfStats.specificity.std.toFixed(4)}</div>
          </div>
          <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "8px" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Mean F1 Score</div>
            <div style={{ fontSize: "1.3rem", fontWeight: "800" }}>{rfStats.f1_score.mean.toFixed(4)}</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>± {rfStats.f1_score.std.toFixed(4)}</div>
          </div>
        </div>

        {/* Repetition Table */}
        <h4 style={{ fontSize: "0.85rem", fontWeight: "700", marginBottom: "8px", color: "var(--text-secondary)" }}>Repetition-by-Repetition Log</h4>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-color)", textAlign: "left", color: "var(--text-muted)" }}>
              <th style={{ padding: "6px" }}>Rep</th>
              <th style={{ padding: "6px" }}>Seed</th>
              <th style={{ padding: "6px" }}>Group</th>
              <th style={{ padding: "6px" }}>ROC-AUC</th>
              <th style={{ padding: "6px" }}>PR-AUC</th>
              <th style={{ padding: "6px" }}>Accuracy</th>
              <th style={{ padding: "6px" }}>Bal Acc</th>
              <th style={{ padding: "6px" }}>Sensitivity</th>
              <th style={{ padding: "6px" }}>Specificity</th>
              <th style={{ padding: "6px" }}>F1</th>
            </tr>
          </thead>
          <tbody>
            {reps.map((r, i) => (
              <tr key={i} style={{ borderBottom: "1px solid var(--border-color)" }}>
                <td style={{ padding: "6px" }}><strong>{r.repetition}</strong></td>
                <td style={{ padding: "6px" }}><code>{r.seed}</code></td>
                <td style={{ padding: "6px" }}>{r.group}</td>
                <td style={{ padding: "6px", fontWeight: "700", color: "var(--brand-blue)" }}>{r.roc_auc.toFixed(4)}</td>
                <td style={{ padding: "6px" }}>{r.pr_auc.toFixed(4)}</td>
                <td style={{ padding: "6px" }}>{r.accuracy.toFixed(4)}</td>
                <td style={{ padding: "6px" }}>{r.balanced_accuracy.toFixed(4)}</td>
                <td style={{ padding: "6px" }}>{r.recall_sensitivity.toFixed(4)}</td>
                <td style={{ padding: "6px" }}>{r.specificity.toFixed(4)}</td>
                <td style={{ padding: "6px" }}>{r.f1_score.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* BLOCK B: LOCKED PEKING_2 TEST */}
      <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid #fde68a", borderRadius: "10px", padding: "20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Lock size={20} color="#b45309" />
            <h3 style={{ fontSize: "1.1rem", fontWeight: "800", color: "#92400e" }}>
              Block B: Locked Held-Out Site Evaluation (Peking_2)
            </h3>
          </div>
          <span style={{ fontSize: "0.75rem", background: "#fef3c7", color: "#b45309", padding: "4px 10px", borderRadius: "9999px", fontWeight: "700" }}>
            Locked Held-out Site: Peking_2 (N=20)
          </span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "20px" }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: "10px" }}>
            <div style={{ background: "#fffbeb", padding: "10px", borderRadius: "6px", border: "1px solid #fef3c7" }}>
              <div style={{ fontSize: "0.7rem", color: "#92400e" }}>Accuracy</div>
              <div style={{ fontSize: "1.25rem", fontWeight: "800", color: "#78350f" }}>{finalTest.accuracy.toFixed(4)}</div>
            </div>
            <div style={{ background: "#fffbeb", padding: "10px", borderRadius: "6px", border: "1px solid #fef3c7" }}>
              <div style={{ fontSize: "0.7rem", color: "#92400e" }}>Balanced Acc</div>
              <div style={{ fontSize: "1.25rem", fontWeight: "800", color: "#78350f" }}>{finalTest.balanced_accuracy.toFixed(4)}</div>
            </div>
            <div style={{ background: "#fffbeb", padding: "10px", borderRadius: "6px", border: "1px solid #fef3c7" }}>
              <div style={{ fontSize: "0.7rem", color: "#92400e" }}>Sensitivity</div>
              <div style={{ fontSize: "1.25rem", fontWeight: "800", color: "#78350f" }}>{finalTest.sensitivity_recall.toFixed(4)}</div>
            </div>
            <div style={{ background: "#fffbeb", padding: "10px", borderRadius: "6px", border: "1px solid #fef3c7" }}>
              <div style={{ fontSize: "0.7rem", color: "#92400e" }}>Specificity</div>
              <div style={{ fontSize: "1.25rem", fontWeight: "800", color: "#78350f" }}>{finalTest.specificity.toFixed(4)}</div>
            </div>
            <div style={{ background: "#fffbeb", padding: "10px", borderRadius: "6px", border: "1px solid #fef3c7" }}>
              <div style={{ fontSize: "0.7rem", color: "#92400e" }}>Precision</div>
              <div style={{ fontSize: "1.25rem", fontWeight: "800", color: "#78350f" }}>{finalTest.precision.toFixed(4)}</div>
            </div>
            <div style={{ background: "#fffbeb", padding: "10px", borderRadius: "6px", border: "1px solid #fef3c7" }}>
              <div style={{ fontSize: "0.7rem", color: "#92400e" }}>F1 Score</div>
              <div style={{ fontSize: "1.25rem", fontWeight: "800", color: "#78350f" }}>{finalTest.f1_score.toFixed(4)}</div>
            </div>
            <div style={{ background: "#fffbeb", padding: "10px", borderRadius: "6px", border: "1px solid #fef3c7" }}>
              <div style={{ fontSize: "0.7rem", color: "#92400e" }}>ROC-AUC</div>
              <div style={{ fontSize: "1.25rem", fontWeight: "800", color: "#78350f" }}>{finalTest.roc_auc.toFixed(4)}</div>
            </div>
            <div style={{ background: "#fffbeb", padding: "10px", borderRadius: "6px", border: "1px solid #fef3c7" }}>
              <div style={{ fontSize: "0.7rem", color: "#92400e" }}>PR-AUC</div>
              <div style={{ fontSize: "1.25rem", fontWeight: "800", color: "#78350f" }}>{finalTest.pr_auc.toFixed(4)}</div>
            </div>
          </div>

          {/* 2x2 Confusion Matrix */}
          <div style={{ background: "#fffbeb", padding: "12px", borderRadius: "8px", border: "1px solid #fde68a" }}>
            <h4 style={{ fontSize: "0.8rem", fontWeight: "700", marginBottom: "8px", color: "#92400e" }}>Peking_2 Confusion Matrix</h4>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px", textAlign: "center" }}>
              <div style={{ background: "#dcfce7", padding: "8px", borderRadius: "4px" }}>
                <div style={{ fontSize: "0.7rem", color: "#166534" }}>True Pos (TP)</div>
                <div style={{ fontSize: "1.2rem", fontWeight: "800", color: "#14532d" }}>{cm.tp}</div>
              </div>
              <div style={{ background: "#fee2e2", padding: "8px", borderRadius: "4px" }}>
                <div style={{ fontSize: "0.7rem", color: "#991b1b" }}>False Neg (FN)</div>
                <div style={{ fontSize: "1.2rem", fontWeight: "800", color: "#7f1d1d" }}>{cm.fn}</div>
              </div>
              <div style={{ background: "#fee2e2", padding: "8px", borderRadius: "4px" }}>
                <div style={{ fontSize: "0.7rem", color: "#991b1b" }}>False Pos (FP)</div>
                <div style={{ fontSize: "1.2rem", fontWeight: "800", color: "#7f1d1d" }}>{cm.fp}</div>
              </div>
              <div style={{ background: "#dcfce7", padding: "8px", borderRadius: "4px" }}>
                <div style={{ fontSize: "0.7rem", color: "#166534" }}>True Neg (TN)</div>
                <div style={{ fontSize: "1.2rem", fontWeight: "800", color: "#14532d" }}>{cm.tn}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
