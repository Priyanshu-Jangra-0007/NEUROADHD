import React from "react";
import { CheckCircle2, ShieldCheck, ArrowRight, Layers } from "lucide-react";

export function ModelView({ data }) {
  const meta = data.final_model_metadata;

  const pipelineSteps = [
    { name: "Raw BOLD Input", tensor: "200 ROIs × T timepoints", desc: "Craddock 200 spatially constrained spectral clustering atlas time courses" },
    { name: "Connectivity Matrix", tensor: "200 × 200 matrix", desc: "Pairwise Pearson correlation computed across all ROI pairs per subject" },
    { name: "Fisher-z Transform", tensor: "19,900 edge vector", desc: "arctanh clipping & transformation of upper-triangle entries (k=1) to stabilize variance" },
    { name: "Median Imputation", tensor: "19,900 edge vector", desc: "SimpleImputer(strategy='median', keep_empty_features=True) fitted inside training folds only" },
    { name: "Variance Filter", tensor: "19,900 edge vector", desc: "VarianceThreshold(threshold=0.0) removes zero-variance constants" },
    { name: "Standardization", tensor: "19,900 edge vector", desc: "StandardScaler() zero-mean unit-variance scaling fitted strictly inside training folds" },
    { name: "Univariate Selection", tensor: "100 selected edges", desc: "SelectKBest(score_func=f_classif, k=100) ANOVA F-test applied strictly on training folds" },
    { name: "Classification Model", tensor: "Scalar probability [0, 1]", desc: "Balanced Random Forest (50 trees, max_depth=3, class_weight='balanced')" }
  ];

  const hyperparams = [
    { param: "K (Selected Edges)", value: "100", note: "Top ANOVA F-score functional connectivity connections (out of 19,900)" },
    { param: "n_estimators", value: "50", note: "Number of decision trees in forest; prevents overfitting on small N" },
    { param: "max_depth", value: "3", note: "Constrained tree depth to limit high-order interaction variance" },
    { param: "min_samples_split", value: "2", note: "Minimum samples required to split an internal decision node" },
    { param: "min_samples_leaf", value: "1", note: "Minimum samples permitted at any individual leaf node" },
    { param: "max_features", value: "'sqrt' (10)", note: "Square root of K: exactly 10 features considered per split candidate" },
    { param: "class_weight", value: "'balanced'", note: "Inverse class-frequency weights to penalize minority ADHD misclassifications" },
    { param: "random_state", value: "42", note: "Fixed deterministic seed for exact scientific reproducibility" }
  ];

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.35rem", fontWeight: "800", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
          Pipeline Architecture &amp; Hyperparameter Specification
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", lineHeight: 1.5, marginTop: "4px" }}>
          Formal specification of the 8-step leakage-safe data transformation pipeline and frozen Balanced Random Forest classifier.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "20px", marginBottom: "24px" }}>
        {/* Pipeline Diagram */}
        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <ShieldCheck size={18} color="var(--accent-success)" />
            <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>
              Leakage-Safe Preprocessing Pipeline
            </h3>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {pipelineSteps.map((step, idx) => {
              const isLast = idx === pipelineSteps.length - 1;
              return (
                <div 
                  key={idx} 
                  style={{ 
                    display: "flex", 
                    alignItems: "flex-start", 
                    gap: "12px", 
                    background: isLast ? "var(--accent-success-subtle)" : "var(--bg-subtle)", 
                    padding: "10px 14px", 
                    borderRadius: "var(--radius-md)", 
                    border: `1px solid ${isLast ? "var(--accent-success)" : "var(--border-color)"}` 
                  }}
                >
                  <span 
                    style={{ 
                      background: isLast ? "var(--accent-success)" : "var(--brand-blue)", 
                      color: "#fff", 
                      width: "22px", 
                      height: "22px", 
                      borderRadius: "var(--radius-full)", 
                      display: "flex", 
                      alignItems: "center", 
                      justifyContent: "center", 
                      fontSize: "0.75rem", 
                      fontWeight: "700", 
                      flexShrink: 0 
                    }}
                  >
                    {idx + 1}
                  </span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: "4px" }}>
                      <span style={{ fontWeight: "700", fontSize: "0.85rem", color: isLast ? "var(--accent-success)" : "var(--text-primary)" }}>
                        {step.name}
                      </span>
                      <span className="tabular-nums" style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                        {step.tensor}
                      </span>
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "2px", lineHeight: 1.4 }}>
                      {step.desc}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Hyperparameters Card */}
        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <CheckCircle2 size={18} color="var(--brand-blue)" />
            <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>
              Classifier Hyperparameters
            </h3>
          </div>

          <div className="table-container" style={{ marginBottom: "16px" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Hyperparameter</th>
                  <th>Value</th>
                  <th>Scientific Rationale</th>
                </tr>
              </thead>
              <tbody>
                {hyperparams.map((hp, i) => (
                  <tr key={i}>
                    <td><code style={{ fontSize: "0.8rem", color: "var(--brand-blue)" }}>{hp.param}</code></td>
                    <td className="tabular-nums" style={{ fontWeight: "700" }}>{hp.value}</td>
                    <td style={{ fontSize: "0.75rem", color: "var(--text-muted)", lineHeight: 1.4 }}>{hp.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ background: "var(--bg-subtle)", padding: "12px 14px", borderRadius: "var(--radius-md)", fontSize: "0.78rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
            <strong>Leakage Guarantee:</strong> All imputer medians, standardization parameters ($\mu, \sigma$), and ANOVA F-statistic rankings are fitted strictly on outer training splits. Test subjects (Peking_2) were never exposed to any preprocessing calculation.
          </div>
        </div>
      </div>
    </div>
  );
}
