import React from "react";
import { GitBranch, ShieldCheck, AlertCircle } from "lucide-react";

export function MethodologyView({ data }) {
  const cohort = data.cohort_breakdown;

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.35rem", fontWeight: "800", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
          Methodology &amp; Dataset Partitioning Protocol
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", lineHeight: 1.5, marginTop: "4px" }}>
          End-to-end data processing workflow, leakage-free cross-validation guarantees, and site-aware evaluation design.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "20px", marginBottom: "24px" }}>
        {/* Partitioning Architecture */}
        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <GitBranch size={18} color="var(--brand-blue)" />
            <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>
              Site-Aware Partitioning Protocol
            </h3>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{ background: "var(--bg-subtle)", padding: "14px", borderRadius: "var(--radius-md)", border: "1px solid var(--border-color)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontWeight: "700", fontSize: "0.85rem" }}>
                <span>1. Training Partition</span>
                <span className="tabular-nums">N = {cohort.train_subjects}</span>
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px", lineHeight: 1.4 }}>
                <span className="tabular-nums">{cohort.train_control}</span> Control, <span className="tabular-nums">{cohort.train_adhd}</span> ADHD across 6 recording sites (KKI, NYU, OHSU, Peking_1, Peking_3, Pittsburgh). Used strictly for internal fold training and cross-validation experiments.
              </div>
            </div>

            <div style={{ background: "var(--bg-subtle)", padding: "14px", borderRadius: "var(--radius-md)", border: "1px solid var(--border-color)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontWeight: "700", fontSize: "0.85rem" }}>
                <span>2. Development Validation Partition</span>
                <span className="tabular-nums">N = {cohort.validation_subjects}</span>
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px", lineHeight: 1.4 }}>
                <span className="tabular-nums">{cohort.validation_control}</span> Control, <span className="tabular-nums">{cohort.validation_adhd}</span> ADHD from external site <code>NeuroIMAGE</code>. Used for intermediate model evaluation and threshold tuning during development.
              </div>
            </div>

            <div style={{ background: "var(--accent-warning-subtle)", padding: "14px", borderRadius: "var(--radius-md)", border: "1px solid rgba(217, 119, 6, 0.3)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontWeight: "700", fontSize: "0.85rem", color: "var(--accent-warning)" }}>
                <span>3. Held-out Peking_2 Batch Partition</span>
                <span className="tabular-nums">N = {cohort.test_subjects}</span>
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--accent-warning)", marginTop: "4px", lineHeight: 1.4 }}>
                10 Control, 10 ADHD from site <code>Peking_2</code>. Held as a batch partition and scored once for this model; the baseline had been scored on the same set earlier. Note: Peking_2 is a same-institution, same-scanner batch holdout (Peking_1 and Peking_3 are in the training set).
              </div>
            </div>
          </div>
        </div>

        {/* Leakage Prevention Rules */}
        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <ShieldCheck size={18} color="var(--accent-success)" />
            <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>
              Strict Leakage-Prevention Safeguards
            </h3>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            <div style={{ background: "var(--bg-subtle)", padding: "12px 14px", borderRadius: "var(--radius-md)", border: "1px solid var(--border-color)", lineHeight: 1.5 }}>
              <strong style={{ color: "var(--text-primary)" }}>Fold-Internal Preprocessing:</strong> Feature selection (SelectKBest ANOVA F-test) and standardization (StandardScaler) are fitted STRICTLY inside outer training folds. No validation or test subjects ever inform feature rankings.
            </div>

            <div style={{ background: "var(--bg-subtle)", padding: "12px 14px", borderRadius: "var(--radius-md)", border: "1px solid var(--border-color)", lineHeight: 1.5 }}>
              <strong style={{ color: "var(--text-primary)" }}>Peking_2 Test Isolation:</strong> No test subjects were ever exposed to imputer median calculations, scaling parameters, or feature selector rankings during training.
            </div>

            <div style={{ background: "var(--bg-subtle)", padding: "12px 14px", borderRadius: "var(--radius-md)", border: "1px solid var(--border-color)", lineHeight: 1.5 }}>
              <strong style={{ color: "var(--text-primary)" }}>Multi-Seed Evaluation Protocol:</strong> Evaluated across 8 independent random seeds (42–49) to document seed variability rather than reporting cherry-picked single splits.
            </div>

            <div style={{ background: "var(--accent-warning-subtle)", padding: "12px 14px", borderRadius: "var(--radius-md)", border: "1px solid rgba(217, 119, 6, 0.3)", lineHeight: 1.5, color: "var(--accent-warning)" }}>
              <strong>Site-Aware Cross-Validation Requirement:</strong> Standard random-fold CV allows subjects from the same site to appear in both train and validation splits. Grouped leave-one-institution-out CV is essential to verify true cross-scanner generalization.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
