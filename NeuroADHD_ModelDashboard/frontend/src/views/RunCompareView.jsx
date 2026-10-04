import React, { useState, useEffect } from "react";
import { GitCompare, CheckCircle2, AlertCircle, ArrowRight, ArrowLeft } from "lucide-react";

export function RunCompareView({ selectedRunIds = [], availableRuns = [], onBack }) {
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (selectedRunIds.length === 0) return;
    async function fetchCompare() {
      try {
        setLoading(true);
        const res = await fetch(`http://localhost:8000/api/runs/compare?run_ids=${selectedRunIds.join(",")}`);
        if (!res.ok) throw new Error("Failed to fetch comparison data");
        const data = await res.json();
        setComparisonData(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchCompare();
  }, [selectedRunIds]);

  if (selectedRunIds.length < 2) {
    return (
      <div className="view-container" style={{ textAlign: "center", padding: "40px 20px" }}>
        <GitCompare size={40} style={{ color: "var(--accent-blue)", marginBottom: "12px" }} />
        <h3>Select Runs to Compare</h3>
        <p style={{ color: "var(--text-secondary)", maxWidth: "500px", margin: "8px auto 20px" }}>
          Go to the <strong>Run History</strong> page and check 2 or more runs to perform a side-by-side comparative analysis of datasets, preprocessing, model hyperparams, and performance metrics.
        </p>
        <button className="btn btn-primary" onClick={onBack}>
          <ArrowLeft size={16} /> Back to Run History
        </button>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="view-container" style={{ textAlign: "center", padding: "40px 20px" }}>
        <div className="loading-spinner" style={{ margin: "0 auto 16px" }} />
        <p style={{ color: "var(--text-secondary)" }}>Comparing {selectedRunIds.length} benchmark runs side-by-side...</p>
      </div>
    );
  }

  if (error || !comparisonData) {
    return (
      <div className="view-container">
        <h3 style={{ color: "var(--accent-red)" }}>Comparison Error</h3>
        <p style={{ color: "var(--text-secondary)" }}>{error || "Unable to compare runs."}</p>
        <button className="btn btn-secondary" onClick={onBack}>Back</button>
      </div>
    );
  }

  const runs = comparisonData.runs || [];

  return (
    <div className="view-container">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div>
          <h2 style={{ fontSize: "20px", fontWeight: "700", marginBottom: "4px" }}>Side-by-Side Run Comparison Matrix</h2>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            Comparing parameters, data checksums, metrics, and environment metadata across {runs.length} selected runs.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={onBack}>
          <ArrowLeft size={15} /> Back to History
        </button>
      </div>

      <div className="table-container" style={{ overflowX: "auto" }}>
        <table className="data-table" style={{ minWidth: "800px" }}>
          <thead>
            <tr>
              <th style={{ width: "200px" }}>Attribute / Metric</th>
              {runs.map((r) => (
                <th key={r.run_id} style={{ textAlign: "center" }}>
                  <code style={{ fontSize: "12px", color: "var(--accent-blue)" }}>{r.run_id}</code>
                  <div style={{ fontSize: "11px", fontWeight: "normal", color: "var(--text-muted)", marginTop: "2px" }}>
                    {r.created_at ? new Date(r.created_at).toLocaleDateString() : ""}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Execution Status</strong></td>
              {runs.map((r) => (
                <td key={r.run_id} style={{ textAlign: "center" }}>
                  <span className={`badge ${r.status === "failed" ? "badge-danger" : "badge-success"}`}>
                    {r.status}
                  </span>
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Selected Best Model</strong></td>
              {runs.map((r) => (
                <td key={r.run_id} style={{ textAlign: "center", fontWeight: "bold", color: "var(--accent-yellow)" }}>
                  {r.selection?.selected_model || "N/A"}
                </td>
              ))}
            </tr>

            <tr style={{ background: "rgba(56, 189, 248, 0.05)" }}>
              <td><strong>Val ROC-AUC (Primary)</strong></td>
              {runs.map((r) => (
                <td key={r.run_id} style={{ textAlign: "center", fontWeight: "bold", fontSize: "15px", color: "var(--accent-green)" }}>
                  {r.selection?.selected_validation_roc_auc ? r.selection.selected_validation_roc_auc.toFixed(4) : "N/A"}
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Val Accuracy</strong></td>
              {runs.map((r) => {
                const bm = r.models?.[r.selection?.selected_model] || {};
                return (
                  <td key={r.run_id} style={{ textAlign: "center" }}>
                    {bm.validation_metrics?.accuracy ? bm.validation_metrics.accuracy.toFixed(4) : "N/A"}
                  </td>
                );
              })}
            </tr>

            <tr>
              <td><strong>Val F1 Score</strong></td>
              {runs.map((r) => {
                const bm = r.models?.[r.selection?.selected_model] || {};
                return (
                  <td key={r.run_id} style={{ textAlign: "center" }}>
                    {bm.validation_metrics?.f1_score ? bm.validation_metrics.f1_score.toFixed(4) : "N/A"}
                  </td>
                );
              })}
            </tr>

            <tr style={{ background: "rgba(250, 204, 21, 0.05)" }}>
              <td><strong>Final Test ROC-AUC</strong></td>
              {runs.map((r) => (
                <td key={r.run_id} style={{ textAlign: "center", fontWeight: "bold" }}>
                  {r.final_test?.metrics?.roc_auc != null ? r.final_test.metrics.roc_auc.toFixed(4) : "Locked"}
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Final Test Accuracy</strong></td>
              {runs.map((r) => (
                <td key={r.run_id} style={{ textAlign: "center" }}>
                  {r.final_test?.metrics?.accuracy != null ? r.final_test.metrics.accuracy.toFixed(4) : "Locked"}
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Dataset SHA-256</strong></td>
              {runs.map((r) => (
                <td key={r.run_id} style={{ textAlign: "center" }}>
                  <code style={{ fontSize: "10px" }}>
                    {r.reproducibility?.feature_table_sha256?.substring(0, 12)}...
                  </code>
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Preprocessing Pipeline</strong></td>
              {runs.map((r) => (
                <td key={r.run_id} style={{ textAlign: "center", fontSize: "12px" }}>
                  SimpleImputer &rarr; Scaler &rarr; SelectKBest(k={r.preprocessing?.k || 100})
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Execution Duration</strong></td>
              {runs.map((r) => (
                <td key={r.run_id} style={{ textAlign: "center", fontSize: "12px" }}>
                  {r.duration_seconds}s
                </td>
              ))}
            </tr>

            <tr>
              <td><strong>Python Version</strong></td>
              {runs.map((r) => (
                <td key={r.run_id} style={{ textAlign: "center", fontSize: "12px" }}>
                  {r.environment?.python_version}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
