import React, { useState } from "react";
import { History, Eye, GitCompare, ExternalLink, ShieldCheck, Clock, CheckCircle2, AlertCircle } from "lucide-react";
import { RunDetailModal } from "./RunDetailModal";

export function RunsHistoryView({ runs, onCompare }) {
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [checkedRuns, setCheckedRuns] = useState([]);
  const [filterText, setFilterText] = useState("");

  const toggleCheck = (runId) => {
    if (checkedRuns.includes(runId)) {
      setCheckedRuns(checkedRuns.filter((id) => id !== runId));
    } else {
      setCheckedRuns([...checkedRuns, runId]);
    }
  };

  const filteredRuns = runs.filter((r) => {
    const q = filterText.toLowerCase();
    return (
      (r.run_id && r.run_id.toLowerCase().includes(q)) ||
      (r.best_model && r.best_model.toLowerCase().includes(q)) ||
      (r.status && r.status.toLowerCase().includes(q))
    );
  });

  return (
    <div className="view-container">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div>
          <h2 style={{ fontSize: "20px", fontWeight: "700", marginBottom: "4px" }}>Permanent Experiment History & Audit Index</h2>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            Every execution produces an immutable run record. Click any run to inspect complete dataset snapshots, hyperparameters, metrics, and logs.
          </p>
        </div>
        {checkedRuns.length > 1 && (
          <button
            className="btn btn-primary"
            onClick={() => onCompare(checkedRuns)}
            style={{ display: "flex", alignItems: "center", gap: "6px" }}
          >
            <GitCompare size={16} /> Compare Selected ({checkedRuns.length})
          </button>
        )}
      </div>

      <div style={{ marginBottom: "16px", display: "flex", gap: "12px", alignItems: "center" }}>
        <input
          type="text"
          placeholder="Filter runs by ID, model, or status..."
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          style={{
            background: "var(--card-bg)",
            border: "1px solid var(--border-color)",
            color: "var(--text-primary)",
            padding: "8px 14px",
            borderRadius: "6px",
            width: "320px",
            fontSize: "13px"
          }}
        />
        <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
          Showing {filteredRuns.length} of {runs.length} historical runs
        </span>
      </div>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width: "40px" }}></th>
              <th>Run ID</th>
              <th>Timestamp (UTC)</th>
              <th>Status</th>
              <th>Best Model</th>
              <th>Val ROC-AUC</th>
              <th>Val Accuracy</th>
              <th>Test ROC-AUC</th>
              <th>Test Accuracy</th>
              <th>Duration</th>
              <th style={{ textAlign: "right" }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredRuns.length === 0 ? (
              <tr>
                <td colSpan="11" style={{ textAlign: "center", padding: "24px", color: "var(--text-muted)" }}>
                  No historical runs found matching your search.
                </td>
              </tr>
            ) : (
              filteredRuns.map((r) => {
                const isChecked = checkedRuns.includes(r.run_id);
                const isFailed = r.status === "failed";
                return (
                  <tr key={r.run_id} className={isChecked ? "selected-row" : ""}>
                    <td>
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => toggleCheck(r.run_id)}
                      />
                    </td>
                    <td>
                      <code style={{ fontSize: "12px", color: "var(--accent-blue)" }}>{r.run_id}</code>
                    </td>
                    <td style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                      {r.created_at ? new Date(r.created_at).toLocaleString() : "N/A"}
                    </td>
                    <td>
                      <span className={`badge ${isFailed ? "badge-danger" : "badge-success"}`}>
                        {isFailed ? <AlertCircle size={12} /> : <CheckCircle2 size={12} />} {r.status}
                      </span>
                    </td>
                    <td>
                      <strong>{r.best_model || "N/A"}</strong>
                    </td>
                    <td>
                      <strong style={{ color: "var(--accent-green)" }}>
                        {r.validation_roc_auc != null ? r.validation_roc_auc.toFixed(4) : "N/A"}
                      </strong>
                    </td>
                    <td>{r.validation_accuracy != null ? r.validation_accuracy.toFixed(4) : "N/A"}</td>
                    <td>
                      <span style={{ color: "var(--accent-yellow)", fontWeight: "bold" }}>
                        {r.final_test_roc_auc != null ? r.final_test_roc_auc.toFixed(4) : "Locked"}
                      </span>
                    </td>
                    <td>{r.final_test_accuracy != null ? r.final_test_accuracy.toFixed(4) : "Locked"}</td>
                    <td style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                      {r.duration_seconds ? `${r.duration_seconds.toFixed(1)}s` : "N/A"}
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <div style={{ display: "flex", gap: "6px", justifyContent: "flex-end" }}>
                        <button
                          className="btn btn-secondary"
                          style={{ padding: "4px 8px", fontSize: "12px" }}
                          onClick={() => setSelectedRunId(r.run_id)}
                          title="View complete run details"
                        >
                          <Eye size={13} /> View
                        </button>
                        <a
                          href={`http://localhost:8000/api/runs/${r.run_id}/html`}
                          target="_blank"
                          rel="noreferrer"
                          className="btn btn-secondary"
                          style={{ padding: "4px 8px", fontSize: "12px", textDecoration: "none" }}
                          title="Open standalone HTML report"
                        >
                          <ExternalLink size={13} /> HTML
                        </a>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {selectedRunId && (
        <RunDetailModal
          runId={selectedRunId}
          onClose={() => setSelectedRunId(null)}
        />
      )}
    </div>
  );
}
