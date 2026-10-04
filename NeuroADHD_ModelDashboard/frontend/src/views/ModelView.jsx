import React from "react";
import { CheckCircle, ShieldCheck } from "lucide-react";

export function ModelView({ data }) {
  const meta = data.final_model_metadata;

  const pipelineSteps = [
    { name: "Raw Input", desc: "CC200 ROI Time Courses (200 functional clusters per subject)" },
    { name: "Connectivity Matrix", desc: "Pairwise Pearson correlation matrix (200 × 200)" },
    { name: "Fisher-z Transform", desc: "arctanh clipping & transformation → 19,900 upper-triangle features" },
    { name: "Imputation", desc: "SimpleImputer(strategy='median', keep_empty_features=True)" },
    { name: "Variance Filter", desc: "VarianceThreshold(threshold=0.0) removes zero-variance constants" },
    { name: "Standardization", desc: "StandardScaler() zero-mean unit-variance scaling" },
    { name: "Feature Selection", desc: "SelectKBest(score_func=f_classif, k=100) on training folds only" },
    { name: "Classifier", desc: "Balanced Random Forest (50 estimators, depth=3, sqrt features)" }
  ];

  const hyperparams = [
    { param: "K (Selected Features)", value: "100", note: "Top ANOVA F-score connectivity edges" },
    { param: "n_estimators", value: "50", note: "Number of decision trees in forest" },
    { param: "max_depth", value: "3", note: "Constrained depth to prevent overfitting on small N" },
    { param: "min_samples_split", value: "2", note: "Minimum samples required to split internal node" },
    { param: "min_samples_leaf", value: "1", note: "Minimum samples required at leaf node" },
    { param: "max_features", value: "'sqrt'", note: "Square root of total selected features (10 features per split)" },
    { param: "class_weight", value: "'balanced'", note: "Inverse class frequency weighting to penalize minority ADHD misclassifications" },
    { param: "random_state", value: "42", note: "Deterministic seed for reproducible evaluation" }
  ];

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: "800", color: "var(--text-primary)" }}>Final Frozen Model Specification</h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          Specification of the selected Balanced Random Forest classifier and leakage-safe preprocessing pipeline.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginBottom: "24px" }}>
        {/* Pipeline Diagram */}
        <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <ShieldCheck size={20} color="var(--accent-green)" />
            <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>Leakage-Safe Preprocessing Pipeline</h3>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {pipelineSteps.map((step, idx) => (
              <div key={idx} style={{ display: "flex", alignItems: "flex-start", gap: "12px", background: idx === pipelineSteps.length - 1 ? "var(--accent-green-bg)" : "var(--bg-subtle)", padding: "10px 14px", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
                <span style={{ background: idx === pipelineSteps.length - 1 ? "var(--accent-green)" : "var(--brand-blue)", color: "#fff", width: "22px", height: "22px", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.75rem", fontWeight: "700", flexShrink: 0 }}>
                  {idx + 1}
                </span>
                <div>
                  <div style={{ fontWeight: "700", fontSize: "0.85rem", color: idx === pipelineSteps.length - 1 ? "var(--accent-green)" : "var(--text-primary)" }}>{step.name}</div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{step.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Hyperparameter Table */}
        <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <CheckCircle size={20} color="var(--brand-blue)" />
            <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>Model Hyperparameters</h3>
          </div>

          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border-color)", textAlign: "left", color: "var(--text-muted)" }}>
                <th style={{ padding: "8px" }}>Parameter</th>
                <th style={{ padding: "8px" }}>Configured Value</th>
                <th style={{ padding: "8px" }}>Design Rationale</th>
              </tr>
            </thead>
            <tbody>
              {hyperparams.map((hp, i) => (
                <tr key={i} style={{ borderBottom: "1px solid var(--border-color)" }}>
                  <td style={{ padding: "8px" }}><code>{hp.param}</code></td>
                  <td style={{ padding: "8px", fontWeight: "700", color: "var(--brand-blue)" }}>{hp.value}</td>
                  <td style={{ padding: "8px", color: "var(--text-muted)", fontSize: "0.75rem" }}>{hp.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
