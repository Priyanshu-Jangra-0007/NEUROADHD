import React from "react";
import { AlertTriangle } from "lucide-react";

export function DisclaimerBanner() {
  return (
    <div style={{
      background: "#fffbeb",
      border: "1px solid #fde68a",
      borderRadius: "8px",
      padding: "10px 16px",
      marginBottom: "20px",
      display: "flex",
      alignItems: "center",
      gap: "10px",
      color: "#92400e",
      fontSize: "0.85rem"
    }}>
      <AlertTriangle size={18} style={{ flexShrink: 0, color: "#d97706" }} />
      <div>
        <strong>Research prototype — not clinically validated.</strong> This platform is designed strictly for machine learning benchmarking, cross-validation robustness, and feature attribution research. Model-derived predictive associations do not constitute biological or clinical diagnoses.
      </div>
    </div>
  );
}
