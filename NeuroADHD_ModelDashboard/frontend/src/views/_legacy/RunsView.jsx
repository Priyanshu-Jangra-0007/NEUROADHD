import React from "react";
import { History, FolderCheck, Clock } from "lucide-react";

export function RunsView({ runs, activeRunId }) {
  return (
    <div className="view-container">
      <h2 style={{ fontSize: "20px", fontWeight: "700", marginBottom: "6px" }}>Benchmark Run History & Audit Logs</h2>
      <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "24px" }}>Inspect past benchmark execution directories, timestamps, and saved joblib model artifacts.</p>

      <div className="table-container">
        <div className="table-header-bar">
          <h3>Completed Benchmark Run Artifacts</h3>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Run Identifier</th>
              <th>Status</th>
              <th>Artifact Paths</th>
              <th>Active Selection</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r, idx) => {
              const isActive = r === activeRunId;
              return (
                <tr key={idx} className={isActive ? "selected-row" : ""}>
                  <td><strong>{r}</strong></td>
                  <td><span style={{ color: "var(--accent-green)", fontWeight: "600" }}>✔ Completed</span></td>
                  <td><code>NeuroADHD_ModelBenchmark/results/{r}</code></td>
                  <td>
                    {isActive ? (
                      <span className="btn btn-secondary" style={{ padding: "2px 8px", fontSize: "11px", pointerEvents: "none" }}>ACTIVE</span>
                    ) : (
                      <span style={{ color: "var(--text-muted)", fontSize: "12px" }}>Archived</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
