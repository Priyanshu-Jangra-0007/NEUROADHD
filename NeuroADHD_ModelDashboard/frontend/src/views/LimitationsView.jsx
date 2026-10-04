import React from "react";
import { AlertTriangle } from "lucide-react";

export function LimitationsView() {
  const limitations = [
    {
      title: "Sample Size Constraints",
      desc: "The current usable cohort consists of 153 subjects across all partitions (114 Train, 19 NeuroIMAGE Validation, 20 Peking_2 Test). While adequate for pilot machine learning benchmarking and cross-validation robustness, larger sample sizes are required for clinical claims."
    },
    {
      title: "Multi-Site Scanner & Acquisition Heterogeneity",
      desc: "fMRI time series originate from multiple sites with distinct scanner hardware, TR acquisition times, and demographic distributions. Performance variability across folds reflects inter-site distributional shifts."
    },
    {
      title: "CC200 Data-Driven Parcellation Representation",
      desc: "The Craddock 200 atlas is constructed via spatial spectral clustering of functional time courses. Node numbers (ROI_001 to ROI_200) represent data-driven clusters without a 1-to-1 anatomical lookup table in local C-PAC release files."
    },
    {
      title: "Non-Causal Interpretability",
      desc: "All SHAP attributions and permutation importance metrics reflect model decision weights in separating fMRI functional connectivity matrices. They do not imply causal neurobiological mechanisms."
    },
    {
      title: "Research Prototype Status",
      desc: "NeuroADHD is strictly an academic research platform for algorithmic exploration and explainability. It is not approved by medical regulatory authorities for diagnostic or treatment planning use."
    }
  ];

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: "800", color: "var(--text-primary)" }}>Study Limitations &amp; Ethical Disclaimers</h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          Explicit boundaries of validity, technical constraints, and appropriate scientific use cases for the NeuroADHD benchmark.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "14px", marginBottom: "24px" }}>
        {limitations.map((lim, i) => (
          <div key={i} className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "8px", padding: "16px", display: "flex", alignItems: "flex-start", gap: "14px" }}>
            <AlertTriangle size={20} color="#d97706" style={{ flexShrink: 0, marginTop: "2px" }} />
            <div>
              <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--text-primary)", marginBottom: "4px" }}>
                {i + 1}. {lim.title}
              </h3>
              <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: "1.5" }}>
                {lim.desc}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
