import React from "react";
import { GitBranch, ShieldCheck } from "lucide-react";

export function MethodologyView({ data }) {
  const cohort = data.cohort_breakdown;

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: "800", color: "var(--text-primary)" }}>Research Methodology &amp; Dataset Partitioning</h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          End-to-end data processing workflow, leakage-free cross-validation guarantees, and independent site isolation rationale.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginBottom: "24px" }}>
        {/* Partitioning Architecture */}
        <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <GitBranch size={20} color="var(--brand-blue)" />
            <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>Site-Aware Data Splitting Protocol</h3>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontWeight: "700", fontSize: "0.85rem" }}>
                <span>1. Training Partition</span>
                <span>N = {cohort.train_subjects}</span>
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
                {cohort.train_control} Control, {cohort.train_adhd} ADHD across 6 recording sites. Used strictly for internal fold training and repeated cross-validation experiments.
              </div>
            </div>

            <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontWeight: "700", fontSize: "0.85rem" }}>
                <span>2. Development Validation Partition</span>
                <span>N = {cohort.validation_subjects}</span>
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
                {cohort.validation_control} Control, {cohort.validation_adhd} ADHD from site <code>NeuroIMAGE</code>. Used for intermediate validation and hyperparameter tuning during development.
              </div>
            </div>

            <div style={{ background: "#fef3c7", padding: "12px", borderRadius: "8px", border: "1px solid #fde68a" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontWeight: "700", fontSize: "0.85rem", color: "#92400e" }}>
                <span>3. Locked Final Test Site</span>
                <span>N = {cohort.test_subjects}</span>
              </div>
              <div style={{ fontSize: "0.75rem", color: "#b45309", marginTop: "4px" }}>
                10 Control, 10 ADHD from independent site <code>Peking_2</code>. Held completely locked and evaluated strictly ONCE after final model freezing to measure out-of-site generalization.
              </div>
            </div>
          </div>
        </div>

        {/* Leakage Prevention Rules */}
        <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <ShieldCheck size={20} color="var(--accent-green)" />
            <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>Strict Leakage-Prevention Safeguards</h3>
          </div>

          <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "10px", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            <li style={{ background: "var(--bg-subtle)", padding: "10px", borderRadius: "6px" }}>
              <strong>Fold-Internal Preprocessing:</strong> Feature selection (SelectKBest ANOVA F-test) and standardization (StandardScaler) are fitted STRICTLY inside outer training folds.
            </li>
            <li style={{ background: "var(--bg-subtle)", padding: "10px", borderRadius: "6px" }}>
              <strong>Zero Test Set Contamination:</strong> No test subjects (Peking_2) were ever exposed to imputer median calculations, scaling parameters, or feature selector rankings during training.
            </li>
            <li style={{ background: "var(--bg-subtle)", padding: "10px", borderRadius: "6px" }}>
              <strong>Repeated Cross-Validation Protocol:</strong> Evaluated across 8 independent random seeds (42–49) to ensure performance claims reflect robust distributions rather than lucky single splits.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
