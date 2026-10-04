import React, { useState } from "react";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";

export function IndividualSubjectView({ data }) {
  const subjects = data.explainability.subject_explanations;
  const [selectedIdx, setSelectedIdx] = useState(0);

  const curSub = subjects[selectedIdx] || subjects[0];

  const posContributors = curSub.top_positive_shap_contributors
    ? curSub.top_positive_shap_contributors.split(" | ")
    : [];

  const negContributors = curSub.top_negative_shap_contributors
    ? curSub.top_negative_shap_contributors.split(" | ")
    : [];

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: "800", color: "var(--text-primary)" }}>Individual Subject Local SHAP Decompositions</h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          Case-by-case feature attribution decomposing Random Forest predictions into positive and negative edge contributions.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "20px", marginBottom: "24px" }}>
        {/* Subject Selector Sidebar */}
        <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "16px" }}>
          <h3 style={{ fontSize: "0.9rem", fontWeight: "700", marginBottom: "12px", color: "var(--text-primary)" }}>
            Select Representative Subject
          </h3>

          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {subjects.map((s, idx) => {
              const isSelected = selectedIdx === idx;
              const isPeking = s.cohort.includes("Peking");
              return (
                <button
                  key={idx}
                  onClick={() => setSelectedIdx(idx)}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "flex-start",
                    padding: "10px 12px",
                    borderRadius: "6px",
                    border: isSelected ? "2px solid var(--brand-blue)" : "1px solid var(--border-color)",
                    background: isSelected ? "var(--brand-blue-bg)" : "var(--bg-subtle)",
                    cursor: "pointer",
                    textAlign: "left"
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", width: "100%", alignItems: "center" }}>
                    <span style={{ fontWeight: "700", fontSize: "0.85rem", color: "var(--text-primary)" }}>
                      Subject {s.subject_id}
                    </span>
                    <span style={{ fontSize: "0.7rem", background: isPeking ? "#fef3c7" : "#e2e8f0", color: isPeking ? "#b45309" : "#475569", padding: "2px 6px", borderRadius: "4px", fontWeight: "600" }}>
                      {s.site}
                    </span>
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "2px" }}>
                    {s.category}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Selected Subject Breakdown */}
        <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "16px" }}>
            <div>
              <span style={{ fontSize: "0.75rem", fontWeight: "700", color: "var(--brand-blue)", textTransform: "uppercase" }}>
                {curSub.cohort} Case Study
              </span>
              <h3 style={{ fontSize: "1.25rem", fontWeight: "800", color: "var(--text-primary)", marginTop: "2px" }}>
                Subject {curSub.subject_id} ({curSub.category})
              </h3>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Predicted ADHD Probability</div>
              <div style={{ fontSize: "1.4rem", fontWeight: "800", color: curSub.prediction_probability_adhd >= 0.5 ? "var(--accent-red)" : "var(--brand-blue)" }}>
                {(curSub.prediction_probability_adhd * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", background: "var(--bg-subtle)", padding: "12px", borderRadius: "8px", marginBottom: "20px" }}>
            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Actual Diagnosis:</span>
              <div style={{ fontWeight: "700", color: curSub.actual_diagnosis === "ADHD" ? "var(--accent-red)" : "var(--brand-blue)" }}>
                {curSub.actual_diagnosis}
              </div>
            </div>
            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Model Classification:</span>
              <div style={{ fontWeight: "700", color: curSub.predicted_class === "ADHD" ? "var(--accent-red)" : "var(--brand-blue)" }}>
                {curSub.predicted_class}
              </div>
            </div>
            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Scanner Site:</span>
              <div style={{ fontWeight: "700", color: "var(--text-primary)" }}>{curSub.site}</div>
            </div>
          </div>

          {/* Attribution Breakdown */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            {/* Positive Contributors (+ADHD) */}
            <div style={{ background: "#fef2f2", border: "1px solid #fee2e2", borderRadius: "8px", padding: "14px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "#991b1b", fontWeight: "700", fontSize: "0.85rem", marginBottom: "8px" }}>
                <ArrowUpRight size={16} />
                <span>Top Positive SHAP Contributors (+ADHD)</span>
              </div>
              <ul style={{ listStyle: "none", padding: 0, margin: 0, fontSize: "0.75rem", color: "#7f1d1d" }}>
                {posContributors.map((c, i) => (
                  <li key={i} style={{ padding: "4px 0", borderBottom: "1px solid #fee2e2" }}>
                    {c}
                  </li>
                ))}
              </ul>
            </div>

            {/* Negative Contributors (+Control) */}
            <div style={{ background: "#eff6ff", border: "1px solid #dbeafe", borderRadius: "8px", padding: "14px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "#1e40af", fontWeight: "700", fontSize: "0.85rem", marginBottom: "8px" }}>
                <ArrowDownRight size={16} />
                <span>Top Negative SHAP Contributors (+Control)</span>
              </div>
              <ul style={{ listStyle: "none", padding: 0, margin: 0, fontSize: "0.75rem", color: "#1e3a8a" }}>
                {negContributors.map((c, i) => (
                  <li key={i} style={{ padding: "4px 0", borderBottom: "1px solid #dbeafe" }}>
                    {c}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
