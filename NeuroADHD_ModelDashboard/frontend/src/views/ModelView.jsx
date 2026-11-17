import React from "react";
import { CheckCircle2, ShieldCheck, ArrowRight, Layers } from "lucide-react";

export function ModelView({ data }) {
  const meta = data?.final_model_metadata;
  const hp = meta?.hyperparameters;
  const kMatch = meta?.feature_selection?.match(/k=(\d+)/);
  const kVal = kMatch ? kMatch[1] : null;

  const pipelineSteps = [
    { name: "Raw BOLD Input", tensor: "200 ROIs × T timepoints", desc: "Craddock 200 spatially constrained spectral clustering atlas time courses" },
    { name: "Connectivity Matrix", tensor: "200 × 200 matrix", desc: "Pairwise Pearson correlation computed across all ROI pairs per subject" },
    { name: "Fisher-z Transform", tensor: "19,900 edge vector", desc: "arctanh clipping & transformation of upper-triangle entries (k=1) to stabilize variance" },
    { name: "Median Imputation", tensor: "19,900 edge vector", desc: "SimpleImputer(strategy='median', keep_empty_features=True) fitted inside training folds only" },
    { name: "Variance Filter", tensor: "19,900 edge vector", desc: "VarianceThreshold(threshold=0.0) removes zero-variance constants" },
    { name: "Standardization", tensor: "19,900 edge vector", desc: "StandardScaler() zero-mean unit-variance scaling fitted strictly inside training folds" },
    { name: "Univariate Selection", tensor: kVal ? `${kVal} selected edges` : "N/A", desc: meta?.feature_selection ? `${meta.feature_selection} ANOVA F-test applied strictly on training folds` : "SelectKBest applied on training folds" },
    { name: "Classification Model", tensor: "Scalar probability [0, 1]", desc: hp ? `Balanced Random Forest (${hp.n_estimators ?? "N/A"} trees, max_depth=${hp.max_depth ?? "N/A"}, class_weight='${hp.class_weight ?? "balanced"}')` : "Balanced Random Forest classifier" }
  ];

  const hyperparams = [
    { param: "K (Selected Edges)", value: kVal ?? "N/A", note: "Top ANOVA F-score functional connectivity connections (out of 19,900)" },
    { param: "n_estimators", value: hp?.n_estimators != null ? String(hp.n_estimators) : "N/A", note: "Number of decision trees in forest; prevents overfitting on small N" },
    { param: "max_depth", value: hp?.max_depth != null ? String(hp.max_depth) : "N/A", note: "Constrained tree depth to limit high-order interaction variance" },
    { param: "min_samples_split", value: hp?.min_samples_split != null ? String(hp.min_samples_split) : "N/A", note: "Minimum samples required to split an internal decision node" },
    { param: "min_samples_leaf", value: hp?.min_samples_leaf != null ? String(hp.min_samples_leaf) : "N/A", note: "Minimum samples permitted at any individual leaf node" },
    { param: "max_features", value: hp?.max_features != null ? (typeof hp.max_features === "string" ? `'${hp.max_features}'` : String(hp.max_features)) : "N/A", note: "Square root of K: features considered per split candidate" },
    { param: "class_weight", value: hp?.class_weight != null ? `'${hp.class_weight}'` : "N/A", note: "Inverse class-frequency weights to penalize minority ADHD misclassifications" },
    { param: "random_state", value: hp?.random_state != null ? String(hp.random_state) : "N/A", note: "Fixed deterministic seed for exact scientific reproducibility" }
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
