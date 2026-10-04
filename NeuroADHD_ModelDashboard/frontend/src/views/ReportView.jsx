import React from "react";
import { ExternalLink, FileText } from "lucide-react";

export function ReportView() {
  const reportUrl = "http://localhost:8000/api/reports/html";

  return (
    <div className="view-container">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div>
          <h2 style={{ fontSize: "20px", fontWeight: "700" }}>Generated HTML Benchmark Report</h2>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>Auto-generated self-contained HTML comparison report from NeuroADHD_ModelBenchmark/reports/model_comparison.html.</p>
        </div>

        <a
          href={reportUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-secondary"
        >
          <ExternalLink size={14} />
          <span>Open Full HTML Report</span>
        </a>
      </div>

      <div style={{ border: "1px solid var(--border-color)", borderRadius: "var(--radius-md)", overflow: "hidden", height: "700px", background: "#ffffff" }}>
        <iframe
          src={reportUrl}
          title="Benchmark Report"
          style={{ width: "100%", height: "100%", border: "none" }}
        />
      </div>
    </div>
  );
}
