import React from "react";
import { Database, Network, Users, Cpu, Lock, CheckCircle2, FileText } from "lucide-react";

export function OverviewView({ data }) {
  const cohort = data.cohort_breakdown;
  const rfStats = data.rf_robustness.summary.all_8_repetitions;
  const finalTest = data.final_test_results;

  const statCards = [
    { title: "Dataset Source", value: "ADHD-200", sub: "C-PAC Preprocessed fMRI", icon: Database, color: "#0284c7" },
    { title: "Parcellation", value: "Craddock 200 (CC200)", sub: "200 Functional Clusters", icon: Network, color: "#8b5cf6" },
    { title: "Cohort Size", value: `${cohort.total_subjects} Subjects`, sub: `${cohort.train_subjects + cohort.validation_subjects} Dev / ${cohort.test_subjects} Peking_2 Test`, icon: Users, color: "#10b981" },
    { title: "Connectivity Matrix", value: "19,900 Edges", sub: "Fisher-z transformed Pearson r", icon: FileText, color: "#f59e0b" },
    { title: "Selected Subnetwork", value: "100 Features", sub: "SelectKBest(f_classif, k=100)", icon: Cpu, color: "#06b6d4" },
    { title: "Final Model", value: "Balanced Random Forest", sub: "Frozen (K=100, depth=3)", icon: CheckCircle2, color: "#16a34a" },
    { title: "Peking_2 Test Site", value: `${cohort.test_subjects} Subjects`, sub: "10 ADHD / 10 Control (Locked)", icon: Lock, color: "#d97706" }
  ];

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: "800", color: "var(--text-primary)" }}>Project Overview</h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          Machine learning pipeline for resting-state fMRI ADHD classification, repeated cross-validation robustness, and TreeSHAP explainability.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "14px", marginBottom: "24px" }}>
        {statCards.map((c, i) => {
          const Icon = c.icon;
          return (
            <div key={i} className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "16px", boxShadow: "var(--shadow-sm)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <span style={{ fontSize: "0.75rem", fontWeight: "600", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{c.title}</span>
                <Icon size={16} color={c.color} />
              </div>
              <div style={{ fontSize: "1.35rem", fontWeight: "800", color: "var(--text-primary)" }}>{c.value}</div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>{c.sub}</div>
            </div>
          );
        })}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginBottom: "24px" }}>
        {/* Development & Split Summary */}
        <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "20px" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: "700", marginBottom: "12px", color: "var(--text-primary)" }}>Cohort &amp; Site Stratification</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border-color)", textAlign: "left", color: "var(--text-muted)" }}>
                <th style={{ padding: "8px" }}>Partition</th>
                <th style={{ padding: "8px" }}>Subjects</th>
                <th style={{ padding: "8px" }}>Control</th>
                <th style={{ padding: "8px" }}>ADHD</th>
                <th style={{ padding: "8px" }}>Sites</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: "1px solid var(--border-color)" }}>
                <td style={{ padding: "8px" }}><strong>Development Train</strong></td>
                <td style={{ padding: "8px" }}>{cohort.train_subjects}</td>
                <td style={{ padding: "8px" }}>{cohort.train_control}</td>
                <td style={{ padding: "8px" }}>{cohort.train_adhd}</td>
                <td style={{ padding: "8px" }}>6 Sites</td>
              </tr>
              <tr style={{ borderBottom: "1px solid var(--border-color)" }}>
                <td style={{ padding: "8px" }}><strong>Development Validation</strong></td>
                <td style={{ padding: "8px" }}>{cohort.validation_subjects}</td>
                <td style={{ padding: "8px" }}>{cohort.validation_control}</td>
                <td style={{ padding: "8px" }}>{cohort.validation_adhd}</td>
                <td style={{ padding: "8px" }}>NeuroIMAGE (1 Site)</td>
              </tr>
              <tr style={{ background: "var(--bg-subtle)" }}>
                <td style={{ padding: "8px" }}><strong>Held-out Final Test</strong></td>
                <td style={{ padding: "8px" }}><strong>{cohort.test_subjects}</strong></td>
                <td style={{ padding: "8px" }}>{cohort.test_control}</td>
                <td style={{ padding: "8px" }}>{cohort.test_adhd}</td>
                <td style={{ padding: "8px" }}><strong>Peking_2 (Locked)</strong></td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Dual Validation Summary */}
        <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "20px" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: "700", marginBottom: "12px", color: "var(--text-primary)" }}>Dual Evaluation Architecture</h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
              <div style={{ fontSize: "0.75rem", fontWeight: "700", color: "var(--brand-blue)", textTransform: "uppercase" }}>8-Repetition CV Mean</div>
              <div style={{ fontSize: "1.4rem", fontWeight: "800", margin: "4px 0" }}>{rfStats.roc_auc.mean.toFixed(4)}</div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>ROC-AUC (± {rfStats.roc_auc.std.toFixed(4)})</div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "4px" }}>Bal Acc: <strong>{rfStats.balanced_accuracy.mean.toFixed(4)}</strong></div>
            </div>
            <div style={{ background: "#fef3c7", padding: "12px", borderRadius: "8px", border: "1px solid #fde68a" }}>
              <div style={{ fontSize: "0.75rem", fontWeight: "700", color: "#b45309", textTransform: "uppercase" }}>Locked Peking_2 Test</div>
              <div style={{ fontSize: "1.4rem", fontWeight: "800", margin: "4px 0", color: "#92400e" }}>{finalTest.metrics.roc_auc.toFixed(4)}</div>
              <div style={{ fontSize: "0.75rem", color: "#b45309" }}>ROC-AUC (Single locked eval)</div>
              <div style={{ fontSize: "0.75rem", color: "#92400e", marginTop: "4px" }}>Bal Acc: <strong>{finalTest.metrics.balanced_accuracy.toFixed(4)}</strong></div>
            </div>
          </div>
          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "12px", lineHeight: "1.4" }}>
            The benchmark clearly distinguishes the repeated 8-fold cross-validation distribution on development data from the one-time locked Peking_2 test set evaluation.
          </p>
        </div>
      </div>
    </div>
  );
}
